try:
    from .deform_conv_2d import DeformConv2D, DeformConvPack2D
    from .deform_conv_2d import ModulatedDeformConv2D, ModulatedDeformConvPack2D
    from .deform_pool_2d import DeformRoIPooling2D, DeformRoIPoolingPack2D
    from .deform_pool_2d import ModulatedDeformRoIPoolingPack2D
    from .deform_conv_3d import DeformConv3D, DeformConvPack3D
    from .deform_conv_3d import ModulatedDeformConv3D, ModulatedDeformConvPack3D
    # from .deform_pool_3d import DeformRoIPooling3D, DeformRoIPoolingPack3D
    # from .deform_pool_3d import ModulatedDeformRoIPoolingPack3D
except:
    print("Warning: DCN modules are not correctly imported!")
