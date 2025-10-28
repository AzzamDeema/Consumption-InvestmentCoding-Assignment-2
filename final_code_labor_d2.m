% INICIALIZE n_function as zeros(300,5)

% (a)
% Clear parameter values, reset.
clear; clc;
rng(1);

% Calibrartion of parameters.
beta=0.96;
gamma=1;
r=0.04;
rho=0.9;
sigma=0.15;
w_bar=2.5; % w steady state
psi=2;
n_ss= 40/168; % n labor hours steady state
omega= w_bar / ((n_ss)^(1/psi)); % omega calibration from policy function

n=5; % Number of states
m=3; %max +- 3 std. devs for normal distribution input in Tauchen function.
mu=w_bar; %unconditional mean of process.

% Define w (5x1) space using Tauchen method ie Z. P is Markov Chain (5x5)
% ie Zprob for good estimation of normal distribution.
[w,P]=Tauchen(n,mu,rho,sigma,m);
%Y=Y(:)'; % Transpose entire column vector of income states to row vector.

% define space for labor
%[n_labor,P]=Tauchen(n,mu,rho,sigma,m);

% Define asset a space.
amin=(-min(w(1))*n_ss)/r; % Natural borrowing limit
amax=100; % Large enough upperlimit for assets (arbitrary).
a=linspace(amin,amax,300); % column --> transpose to row
[a_a, a_aa]=ndgrid(a,a); 
%a=a(:)'; % row
% Equidistant distribution of a with lowerbound amin and upperbound amax of
% 300 points.
% Transpose to convert row vector to column vector of 300x1.

%%%%%%%%%%%%% Novel elements in determining the natural borrowing limit %%%%%%%%%%%%
% The natural limit now consideres that given the labor hours steady state
% of 40/168 hours per week, while earning the lowest possible wage
% -min(w(1)), how much can this household borrow. Rather than the
% -min(exp(Y(1))) from asignment 1, this is replaced with minimum w *
% steady state n.

% Inicialization guess of V0 as more rows than columns. Guess same thing
% for all.
%a_n=300; % Number of assets % Removed for simplicity.
%y_n=5; % Number of income y states % Removed for simplicity.
V0 = zeros(300,5); % Grid space.

% Inicialize V1.
V1=zeros(300,5);

% Inicialize next period assets a'.
a_function=zeros(300,5);

% Inicialize labor hours n_function.
n_function=zeros(300,5);

% Define labor a space.
%nmin=-min(exp(w(1)))/r;
%nmax=100; % Large enough upperlimit for assets (arbitrary).
%n_labor=linspace(nmin,nmax,300); % column --> transpose to row

% Inicialize consumption c function.
c_function=zeros(300,5,300); % c function of a by w states x by a_function a'

% Construct consumption c to search for asset a which maximizes over c an n
% combnations
% objective funcion across asset space of 1 to 300. c = (1+r)a + e^y - a'
% repmat to construct consumption matrix for entire asset a, income y
% state, future asset a' combintations.
% Brute force--> plug in c and search over a range to find max a.
% a' = (300,1)--> repmat --> (300,1*5)= (300,5)
% w' = (1,5)--> repmat --> (1*5,5)= (5,5)
% n_labor'= (300,1) --> repmat --> (300, 1*5) = (300,5)--> transpose --> (5,300)
% w' * n_labor' = (5,5) * (5,300) = (5,300) --> transpose --> (300,5)
% a(a_range) = (1,1)--> repmat --> (1*300,1*5)= (300,5)

%n_function --> (300,5) --> transpose --> (5,300)
%for  a_range=1:300
 %   c(:,:, a_range)=(1+r)*repmat(a',1,5)+(repmat(w',5,1)*(n_function'))'-repmat(a(a_range),300,5);
%end
% Note: Number of income states = 5 on the grid.


% Set inicial tollerance. Note: 10^-9 in instructions. Start small and converge.
e=1e-9;
% Counting condition to prevent an infinity run. Sufficiently large enough 
% maximum number of itterations.
max_iter=10000;

%%%%%%%%%%%%%%%%%%%%% Define utility function with omega %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% NOTICE: Moved to within VFI loop%%%%%%%%%%%%%%%%%%%
% Note: Dr. Ciarliero mentioned must exclude negaive c choices using -inf
% function since negative c will not work with the VFI.

% First, for all negative consumption values, consunmption =0.
%c=max(c,0); % For values c<0, c=0, otherwise c=c

psi_element = 1+(1/psi);
%omega_part = omega*((n_function.^psi_element)/psi_element);

% if environment as not depent on gamma
%u=log(c-omega*((n_labor.^psi_element)/psi_element));
%u(c==0)=-inf; % Utility function for consumption values of zero should be negative infinity.

% u is (300,5,300) 3d matrix, n_function is (300,5) ie (300,5,1) -->
% repmat such that (300,5,300) then dot operation s.t. element wise
% opperations
% log over the entire utility equation!
%u = log(c - omega * repmat(n_function, [1, 1, 300]) .^ psi_element / psi_element);
%u(c==0)=-inf;

%%%%%%%%%%%%%%%%%%%%%%%% VFI %%%%%%%%%%%%%%%%%%%%%%%
% Construct such that more clear definition of w and c current states, computing consumption AFTER VFI look, include  all w states are included
% in V plot

current_iter=0;
norm_V=0.0005;

while (norm_V>e) && (current_iter<max_iter) % && --> AND
    %%%%%%%%%%%%add labor supply policy function iteration step before solving%%%%%%%%%%%%%%%%
    % V_guess
    % use c calculated from before to find labor supply that optimized
    % value function
    % Found by subbing c = (1+r)a + w*n - a') into u(c,n) = long(...)
    % Then, chain rule to find FOC wrt n
    % Suppose inside of long(...) = Z(n)
    % FOC = (1/Z(n)) * derivative Z(n) wrt n --> set = 0
    % Z(n) eliminated
    % Find derivative Z(n) wrt n --> = 0--> solve for n --> n =
    % (w/omega)^psi
    % dot opperation st. element matrix operation for (300,5) matrix n,
    % iterate to update n_function labor hours st. not zero matrix
    n_function = repmat(w',300,1) ./ omega .^ psi;

       for w_state = 1:5
        w_current = w(w_state);
        n_current = n_function(:, w_state);         % 300x1
        n_matrix = n_current * ones(1,300);            % 300x300

        c_construct = (1+r)*a_a + w_current*n_matrix - a_aa;   % c = (1+r)a + w n - a'
        inside_log = c_construct - omega*(n_matrix.^psi_element)/psi_element;

        u_construct = -inf(300,300);
        % ensure u consideres > 0 values before taking log
        restrict = inside_log > 0;
        u_construct(restrict) = log(inside_log(restrict));

        Expectation = V0 * P(w_state,:)';                    % expected value by a'
        v_matrix = u_construct + beta * (ones(300,1) * Expectation');   % add βEV

        [V1(:,w_state), a_function(:,w_state)] = max(v_matrix, [], 2);
       end

  
   norm_V=max(max(abs(abs(V1-V0)))); % V1 is matrix, take double max and abs s.t. take max of 
   % each column, return row of maximum values, then takes maximum value of that row to return
   % scalar of maximum value.
   V0=V1; % Set resulting value V_guess' to V1 and set V1 to origional 
   % guess V0.
   current_iter=current_iter+1;
end

% Compute c policy function given previous VFI.
% Recall: c(:,:, a_range)=(1+r)*repmat(a,1,5)+exp(repmat(Y,300,1))-repmat(a(a_range),300,5);
%c=(1+r)*repmat(a',1,5)+(repmat(w',5,1)*(n_function'))'- a(a_function); % c=300x5,
% n_function = repmat(w',300,1) ./ omega .^ psi; % redundant?

% Compute c AFTER VFI function over w states
c = zeros(300,5);
for w_state = 1:5
    a_construct = a_function(:, w_state);
    c(:, w_state) = (1+r)*a' + w(w_state)*n_function(:, w_state) - a(a_construct)';
end

% Check no longer a matrix of zeros (remove ;)
% n_function;


% (b)
% Plot V(a,w) for all y on one axes.
figure(19); hold on; grid on;
plot(a, V1, 'LineWidth', 1.8);
xlabel('Assets  a'); ylabel('Value  V(a,w)');
title('Converged Value Function by Wage State');
legend(arrayfun(@(i) sprintf('w_{%d}=%.3f', i, w(i)), 1:5, 'UniformOutput', false));
hold off;

% (c) 1000 Simulations.
% epsilon=randn(1000);

% Simulate Y using dtmc(P) which creates a discrete-time, finite-state,
% time-homogeneous Markov chain object. Markov chain is useful here (as
% states in class) because it stands for good estimation of normal
% distribution. Used randn to have normal distribution.
% X = simulate(mc,numSteps) returns data X on random walks of length 
% numSteps (1000 in this case) through sequences of states in the 
% discrete-time Markov chain mc.
P_object=dtmc(P);

T=1000;
drop=500;

simulate_w=simulate(P_object,T); % need Y to compute c and a'
a_inicial=1; % inicializing

%for t=1:T
 %   simulate_n(t)= simulate_w(t)/omega ^ psi;
  %  simulate_c(t)=(1+r)*a(a_inicial(t)) + simulate_w(t)*simulate_n(t) - a(a_function(a_inicial(t), simulate_w(t)));
   % a_inicial(t+1)=a_function(a_inicial(t), simulate_w(t));
   % simulate_a(t+1)=a(a_inicial(t+1)); % sim_a is next period assets
%end

for t=1:T
    w_current = w(simulate_w(t)); % Get actual wage value
    simulate_n(t) = n_function(a_inicial(t), simulate_w(t));
    simulate_c(t) = (1+r)*a(a_inicial(t)) + w_current*simulate_n(t) - a(a_function(a_inicial(t), simulate_w(t)));
    a_inicial(t+1) = a_function(a_inicial(t), simulate_w(t));
    simulate_a(t+1) = a(a_inicial(t+1)); % sim_a is next period assets
end

figure(20)
tiledlayout(4,1, "TileSpacing","compact", "Padding","compact");

nexttile;
plot(drop+1:T, simulate_w(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Income');

nexttile;
plot(drop+1:T, simulate_a(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Next Period Assets');

nexttile;
plot(drop+1:T, simulate_n(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Labor Supply'); 

nexttile;
plot(drop+1:T, simulate_c(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Consumption'); 

% (d)
% std calculates the s.d. of ur simulated c only for the last 500 simulations.
std_c=std(simulate_n(drop+1:T));


%Explain what you would qualitatively expect would occur to the standard 
% deviation of n in the following cases.
 
%(a) The borrowing constraint were zero.
% Borrowing contraint being zero means that household would not be able to borrow assets.
% This would force households to adjust labor when their income stream is
% low, working more. Their labor would thus follow variations in the income
% stream, increasing the stabndard deviation of n labor hours (supply).

%(b) The relative risk aversion parameter doubled.
% Households become more risk averse and will adjust their labor supply,
% working more to smooth consumption. As such, standard deviation of labor
% hours increases as consumers adjust labor frequently to match with income
% volitility.

%(c) The Frisch labor supply elasticity doubled.
% Households become more willing to provide labor given chnages in wage. As
% such, labor supply becomes more sensitive and reactive, increasing
% standard deviation of n.

%(d) Real wage volatility doubled.
% As real wage volitility increases, households will provide more labor to smooth
% consumption. Labour supply tracks income therefor as real wage volitility
% rises, standard deviation of n rises as well.
