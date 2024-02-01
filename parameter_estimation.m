close all
clear all
clc
%model parameters
m=10;%mass
k=5;%spring constant
b=3;%damping coefficient
omega=sqrt(k/m);
gamma=b/m;

%obtaining true trajectory
A=[0 1;-k/m -b/m];%coefficientmatrixforcontinuoussystem
rhs=@(t,x)A*x;%right hand side of the function
xinit=[0;1];%initial value
h=0.01;%time step
T=100;%Final time value
time=0:h:T;%Full time scale
[t,trueTrajectory]=ode45(rhs,time,xinit);
obsNoise=0.1^2; %define observation noise level
obs=trueTrajectory(:,1);%First state is observable
obs=obs+obsNoise*randn(size(obs));%Add noise
%%Extended Kalman Filter
xs=[0;1;1;1];%Initial state estimate(position,velocity,spring constant,damping)
xsEstimate=xs;
P=diag([0.1 0.1 0.1 0.1]);%Initial covariance
varEstimate=diag(P);%Initial state variance
A0=[0 1 0 0;-xs(3)/m -xs(4)/m -xs(1)/m -xs(2)/m;0 0 0 0;0 0 0 0];%linearized continuous 
                                                                  % transition matrix)                   
F=eye(length(xs))+A0*h+(A0.^2*h.^2)/factorial(2)+(A0.^3*h.^3)...
    /factorial(3)+(A0.^4*h.^4)/factorial(4)+(A0.^5*h.^5)/factorial(5)...
    +(A0.^6*h.^6)/factorial(6);%linearized discretized transition matrix
H=[1 0 0 0];%Observation matrix
Q_variance=0.01^2;%process noise variance,a piecewise constant white noise
Q=[1/4*h^4 1/2*h^3 1/2*h^2 1/2*h^2;1/2*h^3 h^2 h h; 1/2*h^2 h 1 1;1/2*h^2 h 1 1]*Q_variance;
R=obsNoise;%Measurement Noise

for i=2:length(obs)
%Prediction step {@a, }
f={@(x1,x2,x3,x4,t)(x1+x2*t);
    @(x1,x2,x3,x4,t)(x2+(-x4/m*x2-x3/m*x1)*t);
    @(x1,x2,x3,x4,t)(x3);
    @(x1,x2,x3,x4,t)(x4)};
xs(1,1)=f{1}(xs(1),xs(2),xs(3),xs(4),h);
xs(2,1)=f{2}(xs(1),xs(2),xs(3),xs(4),h);
xs(3,1)=f{3}(xs(1),xs(2),xs(3),xs(4),h);
xs(4,1)=f{4}(xs(1),xs(2),xs(3),xs(4),h);

A0=[0 1 0 0;-xs(3)/m -xs(4)/m -xs(1)/m -xs(2)/m;0 0 0 0;0 0 0 0];
F=eye(length(xs))+A0*h+(A0^2*h^2)/factorial(2)+(A0^3*h^3)/factorial(3)+...
    (A0^4*h^4)/factorial(4)+(A0^5*h^5)/factorial(5)+(A0^6*h^6)/factorial(6);%State

P=F*P*F'+Q;
%Observation update
K=P*H'*inv(H*P*H'+R);
ee=obs(i)-H*xs;%innovation
xs=xs+K*(ee);
P=P-K*H*P;
xsEstimate(:,i)=xs(:,1);
varEstimate(:,i)=diag(P);
KalmanGain(:,i)=K;
innovation(:,i)=ee;
end


figure(1)
plot(time,trueTrajectory(:,1),'k','linewidth',2)
hold on;
plot(time,obs,'bo','markerfacecolor','b','markersize',3)
plot(time,xsEstimate(1,:),'r','linewidth',3);
axis([0 30,-1 1.5])
grid on
legend 'TrueTrajectory' 'SimulatedNoise' 'KFEstimate'
xlabel('Time(s)')
ylabel('Position(m)')

figure(2)
plot(time,trueTrajectory(:,2),'k','linewidth',3);
hold on;
plot(time,xsEstimate(2,:),'r','linewidth',3);
axis([0 30,-1 1.5])
grid on
legend 'TrueTrajectory' 'KFEstimate'
xlabel('Time(s)')
ylabel('Position(m)')

figure(3)
plot(time,xsEstimate(3,:),'r','linewidth',2);
axis([0 30 ,0 20])
grid on;
legend 'KFEstimate';
xlabel('Time(s)');ylabel('SpringConstant(N/m)')

figure(4)
plot(time,xsEstimate(4,:),'r','linewidth',2);
axis([0 30 ,0 10])
grid on;
legend 'KFEstimate';
xlabel('Time(s)');
ylabel('DampingCoefficient(N/msˆ−1)')
