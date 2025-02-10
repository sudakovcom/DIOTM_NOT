import torch
import torch.nn as nn

def get_model(args):
    if args.model_name in ['ddpm', 'ncsnpp']:
        from .ncsnpp.discriminator import Discriminator_small, Discriminator_large, Discriminator_largest
        from .ncsnpp.ncsnpp_generator_adagn import NCSNpp
        if args.model_name == 'ddpm':
            args.fir = False
            args.resblock_type = 'ddpm'
        
        netG = NCSNpp(args)
        if args.image_size <= 32:
            netD = Discriminator_small(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2))
        elif args.image_size >= 64:
            netD = Discriminator_large(nc = args.num_channels, ngf = args.ngf, act=nn.LeakyReLU(0.2))
        # else:
        #     netD = Discriminator_largest(nc = args.num_channels, ngf = args.ngf, t_emb_dim=args.t_emb_dim, act=nn.LeakyReLU(0.2))

    return netD, netG


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser('UOTM parameters')
    
#     # Experiment description
#     parser.add_argument('--dataset', default='cifar10', choices=['checkerboard', '8gaussian', '25gaussian', 'mnist', 'cifar10', 'celeba64', 'lsun', 'celeba_256'], help='name of dataset')
#     parser.add_argument('--image_size', type=int, default=32, help='size of image (or data)')
#     parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    
#     # Network configurations
#     parser.add_argument('--model_name', default='ncsnpp', choices=['ncsnpp', 'ddpm', 'otm', 'toy'])
#     parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
#     parser.add_argument('--num_channels_dae', type=int, default=128, help='number of initial channels in denoising model')
#     parser.add_argument('--n_mlp', type=int, default=4, help='number of mlp layers for z')
#     parser.add_argument('--ch_mult', nargs='+', type=int, default=[1,2,2,2], help='channel multiplier')
#     parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
#     parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
#     parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
#     parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
#     parser.add_argument('--fir', action='store_false', default=True, help='FIR')
#     parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
#     parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
#     parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
#     parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
#     parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
#     parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
#     parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
#     parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
#     parser.add_argument('--not_use_tanh', action='store_true', default=False, help='use tanh for last layer')
#     parser.add_argument('--z_emb_dim', type=int, default=256, help='embedding dimension of z')
#     parser.add_argument('--nz', type=int, default=100, help='latent dimension')
#     parser.add_argument('--ngf', type=int, default=64, help='The default number of channels of model')
#     parser.add_argument('--num_timesteps', default=2)
#     args = parser.parse_args()

#     netD, netG = get_model(args)
#     print('Succesfully called the models')