import torch
import time
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
import os
from collections import OrderedDict
from utils.utils import print_current_loss
from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler


class EDMTrainer(object):
    def __init__(self, args, model, accelerator, model_ema=None):
        """初始化EDM训练器"""
        self.opt = args
        self.accelerator = accelerator
        self.device = self.accelerator.device
        self.model = model
        self.diffusion_steps = args.diffusion_steps
        self.model_ema = model_ema

        # EDM特有参数
        self.sigma_data = args.sigma_data  # 0.5
        self.P_mean = args.P_mean  # -1.2
        self.P_std = args.P_std  # 1.2
        # 默认值来自diffuser_params.yaml
        self.sigma_min = getattr(args, 'sigma_min', 0.002)
        # 默认值来自diffuser_params.yaml
        self.sigma_max = getattr(args, 'sigma_max', 80.0)

        # 设置TensorBoard
        if self.accelerator.is_main_process:
            starttime = time.strftime("%Y-%m-%d_%H:%M:%S")
            print("Start experiment:", starttime)
            self.writer = SummaryWriter(log_dir=pjoin(
                args.save_root, 'logs_')+starttime[:16], comment=starttime[:16], flush_secs=60)
        self.accelerator.wait_for_everyone()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.98)

    def forward(self, batch_data):
        caption, motions, m_lens = batch_data
        x_start = motions.detach().float()
        # 获取基本信息
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len)
                                   for m_len in m_lens]).to(self.device)
        self.src_mask = self.generate_src_mask(T, cur_len).to(x_start.device)

        # 生成随机噪声水平
        rnd_normal = torch.randn([B], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma[sigma == 0] = self.sigma_min  # 处理sigma为0的情况

        # 计算权重和EDM系数
        self.weight = (sigma ** 2 + self.sigma_data ** 2) / \
            (sigma * self.sigma_data) ** 2
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / \
            ((sigma ** 2 + self.sigma_data ** 2) ** 0.5)
        c_in = 1 / ((sigma**2 + self.sigma_data**2) ** 0.5)
        c_noise = sigma.log() / 4

        # 添加噪声并进行模型预测
        noise = torch.einsum('b,bij->bij', sigma, torch.randn_like(x_start))
        noisy_input = x_start + noise

        # 模型前向传播和预测结果计算
        model_output = self.model(
            torch.einsum('b,bij->bij', c_in, noisy_input),
            c_noise,
            text=caption
        )

        self.target = x_start  # 保存目标值用于损失计算
        self.prediction = torch.einsum('b,bij->bij', c_skip, noisy_input) + \
            torch.einsum('b,bij->bij', c_out, model_output)

    def masked_loss(self, pred, target, mask, weight):
        """计算带掩码的损失
        Args:
            pred: 预测值 [B, T, D]
            target: 目标值 [B, T, D]
            mask: 掩码 [B, T]
            weight: EDM权重 [B]
        Returns:
            loss: 标量损失值
        """
        # 1. 计算MSE并在特征维度上取平均
        squared_error = ((pred - target) ** 2).mean(dim=-1)  # [B, T]

        # 2. 应用mask进行时间维度的加权平均
        loss = (squared_error * mask).sum(-1) / (mask.sum(-1) + 1e-8)  # [B]

        # 3. 应用EDM权重并在batch维度上取平均
        loss = (loss * weight).mean()  # 标量

        return loss

    def backward_G(self):
        """计算反向传播的损失"""
        loss_logs = OrderedDict()
        loss_logs['loss_mot_rec'] = self.masked_loss(
            self.prediction,
            self.target,
            self.src_mask,
            self.weight
        )
        self.loss = loss_logs['loss_mot_rec']
        return loss_logs

    def update(self):
        self.zero_grad([self.optimizer])
        loss_logs = self.backward_G()
        self.accelerator.backward(self.loss)
        self.clip_norm([self.model])
        self.step([self.optimizer])
        return loss_logs

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def train_mode(self):
        self.model.train()
        if self.model_ema:
            self.model_ema.train()

    def eval_mode(self):
        self.model.eval()
        if self.model_ema:
            self.model_ema.eval()

    def save(self, file_name, total_it):
        """保存模型"""
        state = {
            'opt_encoder': self.optimizer.state_dict(),
            'total_it': total_it,
            'encoder': self.accelerator.unwrap_model(self.model).state_dict(),
        }
        if self.model_ema:
            state["model_ema"] = self.accelerator.unwrap_model(
                self.model_ema).module.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        """加载模型"""
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['opt_encoder'])
        if self.model_ema:
            self.model_ema.load_state_dict(
                checkpoint["model_ema"], strict=True)
        self.model.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint.get('total_it', 0)

    def train(self, train_loader):

        it = 0
        if self.opt.is_continue:
            model_path = pjoin(self.opt.model_dir, self.opt.continue_ckpt)
            it = self.load(model_path)
            self.accelerator.print(
                f'continue train from {it} iters in {model_path}')
        start_time = time.time()

        logs = OrderedDict()
        self.dataset = train_loader.dataset
        self.model, self.optimizer, train_loader, self.model_ema = \
            self.accelerator.prepare(
                self.model, self.optimizer, train_loader, self.model_ema)

        num_epochs = (self.opt.num_train_steps-it)//len(train_loader) + 1
        self.accelerator.print(f'need to train for {num_epochs} epochs....')

        for epoch in range(0, num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)
                log_dict = self.update()
                it += 1

                # OneCycleLR修改版本
                # if self.scheduler is not None:
                #     self.scheduler.step()

                if self.model_ema and it % self.opt.model_ema_steps == 0:
                    self.accelerator.unwrap_model(
                        self.model_ema).update_parameters(self.model)

                # 更新日志
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(
                        self.accelerator, start_time, it, mean_loss, epoch, inner_iter=i)
                    if self.accelerator.is_main_process:
                        self.writer.add_scalar(
                            "loss", mean_loss['loss_mot_rec'], it)
                    self.accelerator.wait_for_everyone()

                if it % self.opt.save_interval == 0 and self.accelerator.is_main_process:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)
                self.accelerator.wait_for_everyone()

                if (self.scheduler is not None) and (it % self.opt.update_lr_steps == 0):
                    self.scheduler.step()

        # 保存最后的检查点
        if it % self.opt.save_interval != 0 and self.accelerator.is_main_process:
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), it)

        self.accelerator.wait_for_everyone()
        self.accelerator.print('FINISH')

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    def clip_norm(self, network_list):
        for network in network_list:
            self.accelerator.clip_grad_norm_(
                network.parameters(), self.opt.clip_grad_norm)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()
