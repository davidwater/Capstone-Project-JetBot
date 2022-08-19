class pid ():
    def __init__(self,kp,ki,kd):
        #error cal information
        self.cur=0
        self.desire=0
        #PID parameters
        self.kp=kp
        self.ki=ki
        self.kd=kd
        #error terms
        self.err=0          #error
        self.err_sum=0      #for I ctrl
        self.err_last=0     #for D ctrl
        self.err_dif=0
        #PID output
        self.cmd=0


    def cal_err(self):
        self.err_last=self.err
        self.err=self.desire-self.cur           #cal err
        self.err_sum+=self.err               #integral err
        self.err_dif=self.err-self.err_last  #differential of err


    def output(self):
        self.cmd=(self.kp*self.err)+(self.ki*self.err_sum)+(self.kd*self.err_dif)
        return self.cmd