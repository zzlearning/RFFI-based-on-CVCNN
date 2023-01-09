import numpy as np
def sig_norm(sig,norm_form='maxmin'):
    """ 
    Function:
        将复数信号sig进行归一化/标准化处理
    Arguments:
        sig: ndarry, 复数信号
        norm_form: str, 归一化方式,可选maxmin、z-score
    Return:
        sig: np.ndrray,dtype=np.complex64, 信号序列
     """

    if norm_form=='maxmin':
        real_min=sig.real.min()
        real_max=sig.real.max()
        sig.real=(sig.real-real_min)/(real_max-real_min)

        imag_min=sig.imag.min()
        imag_max=sig.imag.max()
        sig.imag=(sig.imag-imag_min)/(imag_max-imag_min)

        return sig
    elif norm_form=='z_score':
        real_mean=sig.real.mean()
        real_std=sig.real.std()
        sig.real=(sig.real-real_mean)/real_std

        imag_mean=sig.imag.mean()
        imag_std=sig.imag.std()
        sig.imag=(sig.imag-imag_mean)/imag_std

        return sig

sig=np.array([1+1j,2+2j,3+3j])
sig=sig_norm(sig)
print(sig)