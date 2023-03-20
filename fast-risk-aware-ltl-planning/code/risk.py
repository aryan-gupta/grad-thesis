
import img

class Risk:
    def __init__(self, raw_risk_image):
        self.raw_risk_image = raw_risk_image

    # creates tha unknown/assumed risk image
    # pretty much blurs the risk image but will need to find better way to do this
    def create_assumed_risk(self):
        # create blurred risk image
        self.assumed_risk_image = img.create_risk_img(self.raw_risk_image, 32, show=False)
