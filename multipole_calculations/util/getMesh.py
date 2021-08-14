import numpy as np
import os
import trimesh
    
def get_mesh_files(xyzr_filepath, grid_scale=1.0, probe_radius=1.4):
    from multipole_calculations import multipole_calculations_dir_name
    
    xyzr_filename = xyzr_filepath.split("/")[-1]
    xyzr_filename_we = xyzr_filename.split(".")[0]
    nanoshaper_dir_name = os.path.join(multipole_calculations_dir_name, "ExternalSoftware/NanoShaper/")
    mesh_dir_name = os.path.join(nanoshaper_dir_name, "meshs/" + xyzr_filename_we)
    
    if not os.path.exists(nanoshaper_dir_name+"meshs"):
        os.makedirs(nanoshaper_dir_name+"meshs")
        
    if not os.path.exists(mesh_dir_name):
        os.makedirs(mesh_dir_name)
        
    os.system('cp ' + xyzr_filepath + " " + mesh_dir_name)
        
    # Make Changes to Config File
    
    config_template_file = open(nanoshaper_dir_name+'config', 'r')
    config_file = open(nanoshaper_dir_name + 'surfaceConfiguration.prm', 'w')
    
    for line in config_template_file:
        
        if 'XYZR_FileName' in line:
            
            path = os.path.join(mesh_dir_name, xyzr_filename)
            line = 'XYZR_FileName = ' + path + ' \n'
            
        elif 'Grid_scale' in line:
            
            line = 'Grid_scale = {:04.1f} \n'.format(grid_scale)
            
        elif 'Probe_Radius' in line:
            
            line = line = 'Probe_Radius = {:03.1f} \n'.format(probe_radius)
            
        config_file.write(line)
        
    config_file.close()
    config_template_file.close()
    
    os.chdir(nanoshaper_dir_name)
    os.system(nanoshaper_dir_name+"NanoShaper")
    
    os.system('mv ' + nanoshaper_dir_name + '*.vert ' + xyzr_filename_we + '.vert')
    os.system('mv ' + nanoshaper_dir_name + '*.face ' + xyzr_filename_we + '.face')
    
    os.system('mv ' + nanoshaper_dir_name + xyzr_filename_we + '.* ' + mesh_dir_name)
    
    os.chdir(multipole_calculations_dir_name+"/..")
    
def split_mesh(filename):
    
    faces = np.loadtxt(filename+'.face', dtype=int, skiprows=3, usecols=(0,1,2))
    vertices = np.loadtxt(filename+'.vert', dtype=float, skiprows=3, usecols=(0,1,2))
    
    mesh = trimesh.Trimesh(vertices = vertices, faces= faces-1)
    
    mesh_split = mesh.split()
    
    vertices_split = mesh_split[0].vertices
    faces_split = mesh_split[0].faces
    
    os.system("rm " + filename +".vert")
    os.system("rm " + filename +".face")
    
    np.savetxt(filename+'.face', faces_split+1, fmt='%i')
    np.savetxt(filename+'.vert', vertices_split, fmt='%1.5f')