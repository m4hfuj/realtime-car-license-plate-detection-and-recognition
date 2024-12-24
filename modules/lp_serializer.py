import pandas as pd


class LPSerializer:
    def __init__(self):
        self.classes = []
        with open("/home/mahfuj/Projects/OFFICIAL/alpr/classes.txt") as txtFile:
            for line in txtFile:
                self.classes.append(line.strip())

    def serialize(self, bbx):
        license = {'cls':[], 'x2':[], 'y2':[], 'conf':[]}
        if bbx.shape[0] > 0:
            for bb in bbx:
                x2 = int(bb.xywh[0][0].item())
                y2 = int(bb.xywh[0][1].item())
                clsId = int(bb.cls.item())
                # print("ClassID:", clsId)
                clsName = self.classes[clsId]
                conf = bb.conf.item()
                if conf > 0.1:
                    license['x2'].append(x2)
                    license['y2'].append(y2)
                    license['cls'].append(str(clsName))
                    license['conf'].append(conf)


        ## PUT THE TEXT TOGETHER
        lic = pd.DataFrame(license)
        # first the smallest y2 value
        lic = lic.sort_values(by=['y2'])
        # row1
        row_1 = lic.iloc[:-6]
        row_1 = row_1.sort_values(by=['x2'])
        # then exclude first 2 values and sort them by x2
        row_2 = lic.iloc[-6:]
        row_2 = row_2.sort_values(by=['x2'])
        # adding the dataframe together
        lic.iloc[:-6] = row_1
        lic.iloc[-6:] = row_2

        lic = lic.astype(str)
        char_list = lic['cls'].to_list()
        confidence_str_list = lic['conf'].to_list()
        confidence_float_list = [float(num) for num in confidence_str_list]
        # Calculate the average
        if len(confidence_float_list) != 0:
            confidence = sum(confidence_float_list) / len(confidence_float_list)
        else:
            confidence = 0.0


        # print("#################################################\n")
        # print(confidence)
        # print("\n#################################################")

        district = ""
        numbers = ""

        for char in char_list:
            if char.isdigit():
                numbers += str(char)
            else:
                district += str(char)+" "

        # text = ' '.join(lic['cls'])
        # text = str(confidence:.4f) + district + numbers 
        text = f"{district}{numbers}"

        return text, confidence
    