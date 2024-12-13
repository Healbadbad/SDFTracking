# Basic script for running experiment steps
# This is running on windows, otherwise this would be a bash script
import os

if __name__ == "__main__":
    # Generate synthetic data for the cow
    # os.system("python gen_synthetic_data.py --obj_path=data/cow.obj --out_name=cow --n_images=200")

    # Generate Sphere
    # os.system("python gen_synthetic_data.py --out_name=sphere --n_images=40")
    # os.system("python gen_sdf.py --data_name=data/sphere.npz --out_name=sphere")
    # os.system("python pose_estimate_obj.py --sdf_path=./sdf_checkpoints/sphere.pth --vis")


    # Generate Cube
    # os.system("python gen_synthetic_data.py --out_name=cube --n_images=200")
    # os.system("python gen_sdf.py --data_name=data/cube.npz --out_name=cube")
    
    # Train a sdf on the synthetic cow
    # os.system("python gen_sdf.py --data_name=data/cow.npz --out_name=cow")

    # Train a sdf on the lego bulldozer
    # os.system("python -m a4.main --config-name=volsdf")


    # Track the cow
    #os.system("python pose_estimate_obj.py --obj_path=data/cow.obj --sdf_path=./sdf_trained/cow.sdf")
    # os.system("python pose_estimate_obj.py --obj_path=data/cow.obj --sdf_path=./sdf_checkpoints/cow.pth")
    # os.system("python pose_estimate_obj.py --obj_path=data/cow.obj --sdf_path=./sdf_checkpoints/cow.pth --vis")


    # Pack kinect data

    # os.system("python combine_datapack.py --kinect_npz=kinect_records/kinect_data1682542012.6715932.npz --imagestxt_path=kinect_records/exported/images.txt --masks_dir=kinect_records/masks --output_name=data/kinect_mug.npz")

    # Train a sdf on the kinect data
    os.system("python gen_sdf_kinect.py --data_name=data/kinect_mug.npz --out_name=mug")



