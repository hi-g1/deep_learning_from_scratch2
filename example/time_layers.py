import numpy as np

class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False):
        # 배치가 다른 분야의 문장이거나 할때, stateful false로 연관성 없애준대
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers=None

        self.h,self.dh=None,None
        self.stateful=stateful

    def set_state(self,h):
        self.h=h
    
    def reset_state(self):
        self.h=None

    def forward(self,xs):
        Wx,Wh,b=self.params
        N,T,D=xs.shape # N: 미니배치 T: 시퀀스 길이 D: 임베딩 벡터 크기
        D,H=Wx.shape

        self.layers=[]
        hs=np.empty((N,T,H), dtype='f') # 은닉상태 저장용

        if not self.stateful or self.h is None: 
            self.h=np.zeros((N,H), dtype='f') # 최초 호출 시, 초기화
        
        for t in range(T):
            layer=RNN(Wx,Wh,b)
            self.h=layer.forward(xs[:,t,:],self.h)
            hs[:,t,:]=self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx,Wh,b=self.params
        N,T,H=dhs.shape
        D,H=Wx.shape

        dxs=np.empty((N,T,D), dtype='f')
        dh=0
        grads=[0,0,0]
        for t in reversed(range(T)):
            layer=self.layers[t]
            dx,dh=layer.backward(dhs[:,t,:]+dh)
            dxs[:,t,:]=dx

            for i,grad in enumerate(layer.grads):
                grads[i]+=grad # 모든 계층의 그레디언트 저장

        for i,grad in enumerate(grads):
            self.grads[i][...]=grad
        self.dh=dh # stateful에 따라 사용될 수도 있고 아닐 수도 있음

        return dxs

class RNN:
    def __init__(self, Wx, Wh,b):
        self.params=[Wx,Wh,b]
        self.grads=[np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache=None

    def forward(self,x,h_prev):
        Wx,Wh,b=self.params
        t=np.matmul(h_prev,Wh)+np.matmul(x,Wx)+b
        h_next=np.tanh(t)

        self.cache=(x,h_prev,h_next)
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b=self.params
        x,h_prev,h_next=self.cache

        dt=dh_next*(1-h_next**2)
        db=np.sum(dt,axis=0)
        dWh=np.matmul(h_prev.T,dt)
        dh_prev=np.matmul(dt,Wh.T)
        dWx=np.matmul(x.T,dt)
        dx=np.matmul(dt,Wx.T)

        self.grads[0][...]=dWx
        self.grads[1][...]=dWh
        self.grads[2][...]=db

        return dx,dh_prev