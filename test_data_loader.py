from utils.data_loader import DataLoader

loader = DataLoader("dax", "original", 100)
loader.process_all_walks(phase="valid")
loader.process_all_walks(phase="test")