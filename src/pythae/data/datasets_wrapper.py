import torch
from pythae.data.datasets import DatasetOutput

class SIARDatasetWrapper(torch.utils.data.Dataset):
    """ Wrapper class to work with SIAR datasets"""
    
    def __init__(self, data):
        """
        Args:
            data (list or torch Dataset): Iterable where items are dictionaries with keys 'data' and 'label'
        
        """
        self.data = data
        
    def __getitem__(self, index):
        return DatasetOutput(data=self.data[index]['data'], label=self.data[index]['label'], name=self.data[index]['name'])
    
    def __len__(self):
        return len(self.data)
    
class MNISTDatasetWrapper(torch.utils.data.Dataset):
    """ Wrapper class to work with MNIST datasets """
    
    def __init__(self, data):
        """
        Args:
            data (list or torch Dataset): MNIST Dataset
        
        """
        self.data = data
        
    def __getitem__(self, index):
        return DatasetOutput(data=self.data[index][0], label=self.data[index][1])
    
    def __len__(self):
        return len(self.data)
    
class CelebADatasetWrapper(MNISTDatasetWrapper):
    """ Wrapper class to work with CelebA datasets """
    
    def __init__(self, data):
        """
        Args:
            data (list or torch Dataset): MNIST Dataset
        
        """
        super().__init__(data)
