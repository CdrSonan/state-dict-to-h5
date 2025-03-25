import h5py
import torch
import numpy as np

class DictStorage:
    """Class for handling PyTorch state dictionaries and other nested data structures in an HDF5 file"""

    def __init__(self, file, groups:list = [], torchDevice:str = "cpu"):
        """
        file: name of the .hdf5 file
        
        groups: position within the file tree the data is written to. Expected to be a list of strings,
        with each string being the name of a subgroup. Leave empty to save to the root of the file.
        
        torchDevice: the torch.device the tensors will be sent to when loading the model from the file.
        """
        self.group = file
        for i in groups:
            if i not in self.group:
                self.group.create_group(i)
            self.group = self.group[i]
        self.torchDevice = torchDevice
    
    def fetch(self, keys:list):
        """fetch an individual dictionary element."""
        position = self.group
        for i in keys:
            if i not in position:
                raise KeyError("Key not found")
            position = position[i]
        return position[()]
    
    def insert(self, keys:list, value):
        """insert an individual dictionary element."""
        position = self.group
        for i in keys[:-1]:
            if i not in position:
                position.create_group(i)
            position = position[i]
        if keys[-1] in position:
            del position[keys[-1]]
        position.create_dataset(keys[-1], data=value)
        
    def delete(self, keys:list, recursive:bool = False):
        """delete a group from the file, with the option to recursively delete its children."""
        position = self.group
        for i in keys[:-1]:
            if i not in position:
                raise KeyError("Key not found")
            position = position[i]
        if keys[-1] not in position:
            raise KeyError("Key not found")
        del position[keys[-1]]
        if recursive:
            while len(position.keys()) == 0:
                newPosition = position.parent
                del position
                position = newPosition
        
    def toDict(self):
        """deserializes the file and returns the original data structure used during serialization."""
        def unpack(data):
            if data.attrs["type"] == "None":
                return None
            elif data.attrs["type"] in ("float", "int"):
                return data[()].item()
            elif data.attrs["type"] == "bool":
                return bool(data[()].item())
            elif data.attrs["type"] == "str":
                return data[()].decode("utf-8")
            elif data.attrs["type"] == "tensor":
                return torch.tensor(data[()], device=self.torchDevice)
            else:
                print("WARNING: unknown data type hint in loaded file, deserialization may not produce expected results")
                return data[()]
        def recursiveFetch(position):
            if position.attrs["type"] in ("list", "tuple"):
                target = []
                for i in position:
                    if isinstance(position[i], h5py.Group):
                        target.append(recursiveFetch(position[i]))
                    else:
                        target.append(unpack(position[i]))
                if position.attrs["type"] == "tuple":
                    target = tuple(target)
            else:
                target = {}
                for i in position.keys():
                    if i == "__emptyString__":
                        iOut = ""
                    elif i.startswith("__int__"):
                        iOut = int(i[7:])
                    else:
                        iOut = i
                    if isinstance(position[i], h5py.Group):
                        target[iOut] = recursiveFetch(position[i])
                    else:
                        target[iOut] = unpack(position[i])
            
            return target
        return recursiveFetch(self.group)
        
    
    def fromDict(self, dictionary):
        """serializes a dictionary, or other data structure, to the location in the file specified during initialisation."""
        def pack(data):
            if data is None:
                outData = np.array([])
                outDType = "None"
            elif isinstance(data, torch.Tensor):
                outData = data.cpu().numpy()
                outDType = "tensor"
            elif isinstance(data, bool):
                outData = np.array(data)
                outDType = "bool"
            elif isinstance(data, float):
                outData = np.array(data)
                outDType = "float"
            elif isinstance(data, int):
                outData = np.array(data)
                outDType = "int"
            elif isinstance(data, str):
                outData = data.encode("utf-8")
                outDType = "str"
            else:
                raise ValueError("Invalid data type for serialization")
            return outData, outDType
        def recursiveInsert(position, data):
            print(position, data)
            if isinstance(data, dict):
                for key in data.keys():
                    if isinstance(key, int):
                        newKey = "__int__" + str(key)
                    elif isinstance(key, str):
                        if key.startswith("__int__"):
                            raise ValueError("Keys starting with __int__ are reserved for internal use")
                        elif key == "__emptyString__":
                            raise ValueError("__emptyString__ is reserved for internal use")
                        elif key == "":
                            newKey = "__emptyString__"
                        else:
                            newKey = key
                    else:
                        raise ValueError("Keys must be strings or ints")
                    print("key:", key)
                    if isinstance(data[key], dict):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "dict"
                        recursiveInsert(position[newKey], data[key])
                    elif isinstance(data[key], list):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "list"
                        recursiveInsert(position[newKey], data[key])
                    elif isinstance(data[key], tuple):
                        if newKey not in position:
                            position.create_group(newKey)
                            position[newKey].attrs["type"] = "tuple"
                        recursiveInsert(position[newKey], data[key])
                    else:
                        outData, outDType = pack(data[key])
                        position.create_dataset(newKey, data=outData)
                        position[newKey].attrs["type"] = outDType
            elif isinstance(data, list) or isinstance(data, tuple):
                for idx, i in enumerate(data):
                    print("key:", str(idx))
                    if isinstance(i, dict):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "dict"
                        recursiveInsert(position[str(idx)], i)
                    elif isinstance(i, list):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "list"
                        recursiveInsert(position[str(idx)], i)
                    elif isinstance(i, tuple):
                        if str(idx) not in position:
                            position.create_group(str(idx))
                            position[str(idx)].attrs["type"] = "tuple"
                        recursiveInsert(position[str(idx)], i)
                    else:
                        outData, outDType = pack(i)
                        position.create_dataset(str(idx), data = outData)
                        position[str(idx)].attrs["type"] = outDType
        if isinstance(dictionary, dict):
            self.group.attrs["type"] = "dict"
        elif isinstance(dictionary, list):
            self.group.attrs["type"] = "list"
        elif isinstance(dictionary, tuple):
            self.group.attrs["type"] = "tuple"
        recursiveInsert(self.group, dictionary)
