import json

json_files = ["ascend910.json", "cpudram.json", "multigpu.json"]

# Area scaling coefficients (slack_min = C1 * Area, slack_max = C2 * Area)
C1 = 0.01  # 1% of area
C2 = 0.20  # 12% of area

for filename in json_files:
    try:
        with open(filename, "r") as f:
            data = json.load(f)

        for key, info in data.items():
            # Skip metadata keys like "interposer"
            if key == "interposer":
                continue

            if isinstance(info, dict) and "dim" in info:
                width, height = info["dim"]
                area = width * height
                
                slack_min = round(C1 * area, 2)
                slack_max = round(C2 * area, 2)
                
                info["slack"] = [slack_min, slack_max]

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Successfully updated {filename}")

    except FileNotFoundError:
        print(f"File not found: {filename}")
