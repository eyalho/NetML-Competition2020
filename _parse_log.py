with open("records.log", "r") as f:
    for line in f:
        parts = line.split(":")
        if not parts[4].startswith("features"):
            print(parts[2], ",", parts[4], "F1:" + "".join(parts[7:9])[:-3])
