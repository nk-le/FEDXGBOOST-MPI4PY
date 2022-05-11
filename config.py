
test_dataset = "GiveMeSomeCredits"
loss_function = "LogLoss" 


dataset = ["Iris", "GiveMeCredits", "Adult", "DefaultCredits"]
modelArr = ["PlainFedXGBoost", "FedXGBoost"]

CONFIG = {
  "model": modelArr[0],
  "dataset": dataset[2],
}