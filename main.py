from something_dataset import SomethingDataset



dataset = SomethingDataset("something_subset.npz")

sample = dataset[0]
print(sample["image"].shape)
print(sample["text"])          # "Covering [something] with [something]"
print(sample["target"].shape) 