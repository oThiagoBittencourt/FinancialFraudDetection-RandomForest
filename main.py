from NewInstance import new_instance

def instance_inputs():
    step = int(input("Step: "))
    type = str
    while True:
        match int(input("\nType:\n1.CASH-IN\n2.CASH-OUT\n3.DEBIT\n4.PAYMENT\n5.TRANSFER\n\nSelect an option above: ")):
            case 1:
                type = "CASH_IN"
                break
            case 2:
                type = "CASH_OUT"
                break
            case 3:
                type = "DEBIT"
                break
            case 4:
                type = "PAYMENT"
                break
            case 5:
                type = "TRANSFER"
                break
    amount = float(input("Amount: "))
    oldbalanceOrg = float(input("oldbalanceOrg: "))
    newbalanceOrig = float(input("newbalanceOrig: "))
    oldbalanceDest = float(input("oldbalanceDest: "))
    newbalanceDest = float(input("newbalanceDest: "))
    return step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest

while True:
    try:
        print("\n1.Insert a new Instance\n2.Exit\n")
        match int(input("Select an option above: ")):
            case 1:
                step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest = instance_inputs()
                new_instance(step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
            case 2:
                break
    except:
        print("ERROR!")