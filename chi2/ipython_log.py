# Tue, 13 Nov 2012 10:51:18
get_ipython().system(u'ls -F --color ')
# Tue, 13 Nov 2012 10:51:24
import chi2
# Tue, 13 Nov 2012 10:51:25
get_ipython().system(u'ls -F --color ')
# Tue, 13 Nov 2012 10:51:36
get_ipython().magic(u'pinfo chi2.chi2_kernel')
# Tue, 13 Nov 2012 10:51:53
chi2.chi2_kernel(randn(10000,1000))
# Tue, 13 Nov 2012 10:52:05
get_ipython().magic(u'pinfo2 chi2')
# Tue, 13 Nov 2012 10:52:18
chi2.chi2_kernel(randn(10000,1000))
# Tue, 13 Nov 2012 10:52:39
get_ipython().magic(u'pinfo2 chi2.chi2_kernel')
# Tue, 13 Nov 2012 10:52:53
get_ipython().magic(u'pinfo2 chi2.chi2_dist')
# Tue, 13 Nov 2012 10:53:11
chi2.chi2_kernel(randn(10000,1000))
# Tue, 13 Nov 2012 10:53:18
get_ipython().magic(u'pinfo2 chi2.chi2_dist')
# Tue, 13 Nov 2012 10:53:31
chi2.chi2_kernel(randn(10000,1000))
# Tue, 13 Nov 2012 10:53:36
get_ipython().magic(u'pinfo2 chi2.chi2_dist')
# Tue, 13 Nov 2012 10:55:51
import chi2
# Tue, 13 Nov 2012 10:55:59
chi2.chi2_kernel(randn(10000,1000))
# Tue, 13 Nov 2012 10:57:05
import chi2
# Tue, 13 Nov 2012 10:57:29
K = chi2.chi2_kernel(randn(1000,1000).astype(float32))
# Tue, 13 Nov 2012 10:57:49
K = chi2.chi2_kernel((randn(1000,1000)+1).astype(float32))
# Tue, 13 Nov 2012 10:57:53
K
#[Out]# (array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        ..., 
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.]]),
#[Out]#  inf)
# Tue, 13 Nov 2012 10:57:57
K.min()
# Tue, 13 Nov 2012 10:58:12
K = chi2.chi2_kernel((randn(1000,1000)*0.1).astype(float32))
# Tue, 13 Nov 2012 10:58:13
K.min()
# Tue, 13 Nov 2012 10:58:20
K
#[Out]# (array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        ..., 
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.]]),
#[Out]#  inf)
# Tue, 13 Nov 2012 10:58:24
K[0].min()
#[Out]# nan
# Tue, 13 Nov 2012 10:58:26
K[0].max()
#[Out]# nan
# Tue, 13 Nov 2012 10:58:44
data = np.load('/home/vincent/data/hollywood2_unlabeled_traindata_whitened.npy', 'r')[:20000]
# Tue, 13 Nov 2012 10:58:49
data.shape
#[Out]# (20000, 2520)
# Tue, 13 Nov 2012 10:59:00
data = array(np.load('/home/vincent/data/hollywood2_unlabeled_traindata_whitened.npy', 'r')[:20000])
# Tue, 13 Nov 2012 10:59:15
K = chi2.chi2_kernel(data)
# Tue, 13 Nov 2012 11:07:01
data = np.load('/home/vincent/data/hollywood2_unlabeled_traindata_whitened.npy', 'r')[:20000]
# Tue, 13 Nov 2012 11:07:06
import chi2
# Tue, 13 Nov 2012 11:07:08
K = chi2.chi2_kernel(data)
# Tue, 13 Nov 2012 11:10:36
import chi2
# Tue, 13 Nov 2012 11:10:47
chi2.chi2_kernel(randn(10000,1000).astype(float32))
# Tue, 13 Nov 2012 11:12:37
import chi2
# Tue, 13 Nov 2012 11:12:40
data = np.load('/home/vincent/data/hollywood2_unlabeled_traindata_whitened.npy', 'r')[:20000]
# Tue, 13 Nov 2012 11:12:43
K = chi2.chi2_kernel(data)
# Tue, 13 Nov 2012 11:15:35
K
#[Out]# (array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        ..., 
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.],
#[Out]#        [ 1.,  1.,  1., ...,  1.,  1.,  1.]]),
#[Out]#  inf)
# Tue, 13 Nov 2012 11:17:27
data -= data.min()
# Tue, 13 Nov 2012 11:17:43
data = array(data)-data.min()
# Tue, 13 Nov 2012 11:17:46
data.shape
#[Out]# (20000, 2520)
# Tue, 13 Nov 2012 11:17:48
data.min()
#[Out]# memmap(0.0, dtype=float32)
# Tue, 13 Nov 2012 11:17:56
data = array(dat)
# Tue, 13 Nov 2012 11:17:58
data = array(data)
# Tue, 13 Nov 2012 11:18:00
data.min()
#[Out]# 0.0
# Tue, 13 Nov 2012 11:18:05
K = chi2.chi2_kernel(data)
# Tue, 13 Nov 2012 11:22:35
K
#[Out]# (array([[ 1.        ,  0.60972208,  0.6217739 , ...,  0.63551611,
#[Out]#          0.36642486,  0.51989579],
#[Out]#        [ 0.60972208,  1.        ,  0.74184096, ...,  0.74365568,
#[Out]#          0.42532429,  0.61949784],
#[Out]#        [ 0.6217739 ,  0.74184096,  1.        , ...,  0.76503295,
#[Out]#          0.43441123,  0.62993717],
#[Out]#        ..., 
#[Out]#        [ 0.63551611,  0.74365568,  0.76503295, ...,  1.        ,
#[Out]#          0.45003748,  0.64066857],
#[Out]#        [ 0.36642486,  0.42532429,  0.43441123, ...,  0.45003748,
#[Out]#          1.        ,  0.37360099],
#[Out]#        [ 0.51989579,  0.61949784,  0.62993717, ...,  0.64066857,
#[Out]#          0.37360099,  1.        ]], dtype=float32),
#[Out]#  49.89950942993164)
