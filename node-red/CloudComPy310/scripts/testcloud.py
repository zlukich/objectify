import cloudComPy as cc
import os

cloud1 = cc.loadPointCloud("nerf_pc_high_res.ply")
cloud1.setName("cloud1")

cloud2ref = cc.loadPointCloud("ngp_phone_pc_lowres.ply")
cloud2ref.setName("cloud2_reference")





data = cc.loadPointCloud("clouds2.bin")

res=cc.ICP(data=cloud2ref, model=cloud1, minRMSDecrease=1.e-2,
           maxIterationCount=20000, randomSamplingLimit=50000,
           method=cc.CONVERGENCE_TYPE.MAX_ITER_CONVERGENCE,
           removeFarthestPoints = True,
           adjustScale=True, finalOverlapRatio=0.1)
tr2 = res.transMat
cloud3 = res.aligned
cloud3.applyRigidTransformation(tr2)
cloud3.setName("cloud2_transformed_afterICP")

cc.SaveEntities([cloud1, cloud2ref, cloud3], os.path.join(".", "clouds_aligned.bin"))