import os
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ['CFLAGS'] = '-Wno-deprecated-declarations'  # suppress warnings in debug mode


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


def make_cuda_ext(name, module, sources):
    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
    else:
        raise EnvironmentError('CUDA is required to compile RODNet!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':
    setup(
        name='rodnet',
        version='1.3',
        description='RODNet: Object Detection from Radar Data',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yizhou-wang/RODNet',
        author='Yizhou Wang',
        author_email='ywang26@uw.edu',

        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',

            # Pick your license as you wish
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            # These classifiers are *not* checked by 'pip install'. See instead
            # 'python_requires' below.
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        keywords='rodnet, object detection, radar, autonomous driving',

        packages=find_packages(include=["rodnet.*"]),
        # package_data={'rodnet.ops': ['*/*.so']},
        python_requires='>=3.6',
        install_requires=get_requirements(),
        # ext_modules=[
        #     make_cuda_ext(
        #         name='deform_conv_2d_cuda',
        #         module='rodnet.ops.dcn',
        #         sources=[
        #             'src/deform_conv_2d_cuda.cpp',
        #             'src/deform_conv_2d_cuda_kernel.cu'
        #         ]),
        #     make_cuda_ext(
        #         name='deform_conv_3d_cuda',
        #         module='rodnet.ops.dcn',
        #         sources=[
        #             'src/deform_conv_3d_cuda.cpp',
        #             'src/deform_conv_3d_cuda_kernel.cu'
        #         ]),
        #     make_cuda_ext(
        #         name='deform_pool_2d_cuda',
        #         module='rodnet.ops.dcn',
        #         sources=[
        #             'src/deform_pool_2d_cuda.cpp',
        #             'src/deform_pool_2d_cuda_kernel.cu'
        #         ]),
        #     make_cuda_ext(
        #         name='deform_pool_3d_cuda',
        #         module='rodnet.ops.dcn',
        #         sources=[
        #             'src/deform_pool_3d_cuda.cpp',
        #             'src/deform_pool_3d_cuda_kernel.cu'
        #         ]),
        # ],
        # cmdclass={'build_ext': BuildExtension},
        zip_safe=False
    )
