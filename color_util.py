import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ColorTransform(nn.Module):
    """Polynomial color transformation for realistic camouflage patterns.
    تبدیل رنگ چندجمله‌ای برای الگوهای استتار واقع‌گرایانه.
    
    Implements polynomial feature transformation for color mapping.
    تبدیل ویژگی چندجمله‌ای را برای نگاشت رنگ پیاده‌سازی می‌کند.
    """
    def __init__(self, para_path):
        """
        Load polynomial transformation parameters from file.
        بارگذاری پارامترهای تبدیل چندجمله‌ای از فایل.
        
        :param para_path: Path to .npz file with transformation parameters
        :param para_path: مسیر فایل .npz با پارامترهای تبدیل
        """
        super(ColorTransform, self).__init__()
        file = np.load(para_path, allow_pickle=True)
        self.degree = file['d']  # Polynomial degree / درجه چندجمله‌ای
        weight = torch.from_numpy(file['weight'])
        bias = torch.from_numpy(file['bias'])
        # Register as buffers (non-trainable) / ثبت به عنوان بافر (غیرقابل آموزش)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def poly_feature(self, x, degree=None):
        """Generate polynomial features up to specified degree.
        تولید ویژگی‌های چندجمله‌ای تا درجه مشخص.
        
        Creates polynomial combinations of input features (e.g., x^2, x*y, etc.).
        ترکیبات چندجمله‌ای ویژگی‌های ورودی را ایجاد می‌کند.
        
        :param x: Input features / ویژگی‌های ورودی
        :param degree: Polynomial degree / درجه چندجمله‌ای
        :return: Polynomial features / ویژگی‌های چندجمله‌ای
        """
        if degree is None:
            degree = self.degree
        n = x.shape[1]
        feature = [x.clone()]
        index = list(range(n))
        # Build polynomial terms iteratively / ساخت ترم‌های چندجمله‌ای به صورت تکراری
        for d in range(1, degree):
            new = []
            k = 0
            for i in range(n):
                new.append(x[:, i:i + 1] * feature[-1][:, index[i]:])
                index[i] = k
                k = k + new[-1].shape[1]
            new = torch.cat(new, 1)
            feature.append(new)
        feature = torch.cat(feature, 1)
        return feature

    def forward(self, x):
        """Apply polynomial color transformation.
        اعمال تبدیل رنگ چندجمله‌ای.
        
        :param x: Input RGB colors / رنگ‌های RGB ورودی
        :return: Transformed colors / رنگ‌های تبدیل شده
        """
        # Generate polynomial features / تولید ویژگی‌های چندجمله‌ای
        f = self.poly_feature(x)
        f = f.transpose(1, -1)
        # Linear transformation: pred = f * weight + bias
        # تبدیل خطی: pred = f * weight + bias
        pred = torch.matmul(f, self.weight) + self.bias
        pred = pred.transpose(1, -1)
        return pred