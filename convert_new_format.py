import h5py
import numpy as np

def migrate_2d_to_3d_file(old_filename, new_filename):
    # Mapping old names to new names
    rename_map = {
        't': 'te',
        's': 'sa',
        'u': 'u_0',
        'w': 'u_1'
    }

    with h5py.File(old_filename, 'r') as old_f, h5py.File(new_filename, 'w') as new_f:
        print(f"Old keys detected: {list(old_f.keys())}")
        
        for key in old_f.keys():
            # 1. Handle Grid separately (usually kept as xg, zg)
            if key in ['xg', 'zg']:
                new_f.create_dataset(key, data=old_f[key][:])
                continue
                
            obj = old_f[key]
            
            # Determine the target name
            new_key = rename_map.get(key, key)

            # 2. Handle Groups (like 'params')
            if isinstance(obj, h5py.Group):
                print(f"Copying group: {key} -> {new_key}")
                old_f.copy(obj, new_f, name=new_key)
            
            # 3. Handle Datasets (Scalars and Arrays)
            elif isinstance(obj, h5py.Dataset):
                if obj.shape == ():
                    val = obj[()]
                    print(f"Copying scalar: {key} -> {new_key} (Value: {val})")
                else:
                    val = obj[:]
                    print(f"Copying array: {key} -> {new_key}")
                
                new_f.create_dataset(new_key, data=val)

        # 4. Finalizing Metadata
        new_f.attrs['dim'] = 2 
        
    print(f"\nMigration complete! New file '{new_filename}' is ready for the library.")

# Execute
migrate_2d_to_3d_file('RPO2_1_Lx0.8.h5', 'RPO2_1_Lx0.8_new.h5')