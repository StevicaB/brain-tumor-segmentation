from skull import SkullStripper

image_path = ["/home/stevica/Schreibtisch/IDP/Data/REMBRANDT/HF0855/T1.nii", "/home/stevica/Schreibtisch/IDP/Data/REMBRANDT/HF0855/T1c.nii","/home/stevica/Schreibtisch/IDP/Data/REMBRANDT/HF0855/T2.nii", "/home/stevica/Schreibtisch/IDP/Data/REMBRANDT/HF0855/FLAIR.nii"]

stripper = SkullStripper()
stripper.strip_skull(image_path, 0)
