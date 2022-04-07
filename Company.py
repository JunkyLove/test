class Company:
    def __init__(self):
        self.year=None
        self.name=None
        self.code=None
        self.ST=None
        #self.ST_year
        self.characteristics=None
    def get_year(self,year):
        self.year=year
        if self.year!=(int)(-2) and self.year!=(int)(-3):
            raise Exception('请于样本集名称中标注t-2或t-3年!')
        self.identify_characteristics()
    def get_name(self,name):
        self.name=name
    def get_code(self,code):
        self.code=code
    def get_ST(self,flag):
        self.ST=flag


    def identify_characteristics(self):
        if self.year==-3:
            self.characteristics=[0.0]*6
        else:
            self.characteristics=[0.0]*7




