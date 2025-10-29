% INICIALIZE n_function as zeros(300,5)

% (a)
% Clear parameter values, reset.
clear; clc;
rng(1);

% Calibrartion of parameters.
%Wage Calibration
beta=0.96;
%gamma=1;
%r=0.04;
rho=0.9;
sigma=0.15;
w_bar=2.5; % w steady state

% Capital Calibration
alpha=0.4;
delta_0=0.1;
phi_1=(1/beta)-(1-delta_0);
phi_2=0.2;


n=5; % Number of states
m=3; %max +- 3 std. devs for normal distribution input in Tauchen function.
mu=w_bar; %unconditional mean of process.

% Define w (5x1) space using Tauchen method ie Z. P is Markov Chain (5x5)
% ie Zprob for good estimation of normal distribution.
[w,P]=Tauchen(n,mu,rho,sigma,m);
%Y=Y(:)'; % Transpose entire column vector of income states to row vector.

% define space for labor
%[n_labor,P]=Tauchen(n,mu,rho,sigma,m);

% Define asset k space.
kmin=0.1;
kmax=100; % Large enough upperlimit for assets (arbitrary).
k=linspace(kmin,kmax,300); % column --> transpose to row
%a=a(:)'; % row
% Equidistant distribution of a with lowerbound amin and upperbound amax of
% 300 points.
% Transpose to convert row vector to column vector of 300x1.


% Inicialization guess of V0 as more rows than columns. Guess same thing
% for all.
%a_n=300; % Number of assets % Removed for simplicity.
%y_n=5; % Number of income y states % Removed for simplicity.
V0 = zeros(300,5); % Grid space.
% Inicialize V1.
V1=zeros(300,5);
% Inicialize next period capital k'
k_function=zeros(300,5);
% Inicialize labor hours n_function
n_function=zeros(300,5);
% Inicialize capital utilization u_function
u_function=zeros(300,5);
% Inicialize investment inv_function
inv_function=zeros(300,5);

% Set inicial tollerance. Note: 10^-9 in instructions. Start small and converge.
e=1e-9;
% Counting condition to prevent an infinity run. Sufficiently large enough 
max_iter=10000; % maximum number of itterations.
current_iter=0;
norm_V=0.0005;


% Exclude zeros
inv_function = max(inv_function, 0);

% Inicialization with zeros, three dimensions
production_fun = zeros(300, 5, 300);
inv_prime = zeros(300, 5, 300);
wn = zeros(300, 5, 300);


while (norm_V>e) && (current_iter<max_iter)
    %%%%%%%%%%%%add labor supply policy function iteration step before solving%%%%%%%%%%%%%%%%
    % dot product ie element multiplication not full on matrix
    % u_function = (300,5)
    % k' = (300,1) --> repmat (300, 5)
    % u_function * k' = (300,5) because element multiplication!
    % w' = (1,5) --> repmat --> 300,5
    % u_function * k' / w' = (300,5) = n_function
    % n_function = ((1-alpha) * repmat(u_function, 1, 1) .* repmat(k', 1,5) .^(alpha) ./ repmat(w',300,1)) .^(1/alpha);
    % break it down into pieces
    uk_matrix = (u_function .* repmat(k', 1, 5)).^ alpha; % (300x5)
    w_matrix = repmat(w', 300, 1); % (300x5)
    n_function = ((1-alpha) * uk_matrix ./ w_matrix) .^ (1/alpha);


    % Define functions
    for k_range = 1:300
    % Production: f(uk,n) = (uk)^alpha * n^(1-alpha)
    % u_function --> 1,5 --> repmat --> 300,5
    % k' = 300,1 --> repmat --> 300,5
    % dot product ie element multiplication not full on matrix
    % production_fun(:,:,k_range) = (repmat(u_function(k_range,:), 300, 1) .* repmat(k', 1, 5)) .^ alpha .* repmat(n_function(k_range,:), 300, 1) .^ (1-alpha);
    % break it down
    uk_current = (u_function(k_range,:) .* repmat(k', 1, 5)).^ alpha; % Element-wise multiplication (1x5)
    n_repmat = repmat(n_function(k_range,:), 300, 1); % (300x5)
    production_fun(:,:,k_range) = uk_current .* n_repmat .^ (1-alpha);

    %delta_u = 300,5
    delta_u = delta_0 + phi_1 * (repmat(u_function(k_range,:), 300, 1) - 1) + phi_2/2 * (repmat(u_function(k_range,:), 300, 1) - 1).^2;

    % Investment process k′ = [1−δ(u)]k +inv′
    % inv_prime = 300,5
    % (1 - delta_u) = (300,5) matrix where each ELEMENT is 1 - delta_u( , )
    inv_prime(:,:,k_range) = repmat(k',1,5) - (1 - delta_u) .* repmat(k(k_range),300,5);
    
    % IMPOSE inv' >= 0
    inv_prime(:,:,k_range) = max(inv_prime(:,:,k_range),0);
    
    % w*n --> 300,5
    % w'=1,5 --> repmat --> 300,5
    % n_function = 300,5 but n_function(k_range)= 1,5 --> repmat --> 300,5
    % for 1 to 300
    wn(:,:,k_range)= repmat(w', 300, 1) .* repmat(n_function(k_range,:), 300, 1);
    end


   
 V_guess=zeros(300,300); % Iniciallization, current a* choice future a'.
   for w_state=1:5
           Expectation_v = V0 * P(w_state,:)';    % 300×1
            for k_range = 1:300
                    V_guess(:,k_range) = production_fun(:,w_state,k_range) ...
                                   - inv_prime(:,w_state,k_range) ...
                                   - wn(:,w_state,k_range) ...
                                   + beta * Expectation_v(k_range);
            end
       [V1(:,w_state),k_function(:,w_state)]=max(V_guess');
   end

   norm_V=max(max(abs(abs(V1-V0)))); % V1 is matrix, take double max and abs s.t. take max of 
   % each column, return row of maximum values, then takes maximum value of that row to return
   % scalar of maximum value.
   V0=V1; % Set resulting value V_guess' to V1 and set V1 to origional 
   % guess V0.
   current_iter=current_iter+1;

   u_function = ones(300, 5) + 0.1 * repmat(k', 1, 5) / max(k);
    delta_u_policy = delta_0 + phi_1*(u_function - 1) + (phi_2/2)*(u_function - 1).^2;  % Nk x n
    inv_function = k(k_function) - (1 - delta_u) .* repmat(k', 1, 5);
    n_function = repmat(((1-alpha) ./ w').^(1/alpha), 300, 1) .* repmat(k', 1, 5);
    

end

% Policy functions
%inv_function = k(k_function) - (1 - delta_u) .* repmat(k', 1, 5);
%n_function = repmat(((1-alpha) ./ w').^(1/alpha), 300, 1) .* repmat(k', 1, 5);
%u_function = ones(300, 5) + 0.1 * repmat(k', 1, 5) / max(k);
  


% (b) Graph the converged value function in (k,w) space for all w.

% (b) Plot V(k,w) for all w
figure(10);
hold on; grid on;
plot(k, V1, 'LineWidth', 1.8);
xlabel('Capital k');
ylabel('Value V(k,w)');
title('Converged Value Function by Wage State');
hold off;

% (c)

% (c) 1000 Simulations
P_object = dtmc(P);

T = 1000;
drop = 500;

simulate_w = simulate(P_object, T); % need w to compute k', n, inv', u
k_inicial = 1; % inicializing

for t = 1:T
    k_now = k_inicial(t);
    w_now = simulate_w(t);
    
    % Get policy functions for current state (using indices)
    simulate_n(t) = n_function(k_now, w_now);
    simulate_u(t) = u_function(k_now, w_now);
    simulate_inv(t) = inv_function(k_now, w_now);
    k_inicial(t+1) = k_function(k_now, w_now);
    simulate_k(t+1) = k(k_inicial(t+1));
end

figure(20)
tiledlayout(5,1, "TileSpacing","compact", "Padding","compact");

nexttile;
plot(drop+1:T, w(simulate_w(drop+1:T)), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Wage');

nexttile;
plot(drop+1:T, simulate_k(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Next Period Capital');

nexttile;
plot(drop+1:T, simulate_n(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Labor Demand'); 

nexttile;
plot(drop+1:T, simulate_inv(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Investment'); 

nexttile;
plot(drop+1:T, simulate_u(drop+1:T), 'LineWidth', 1.5); 
grid on;
xlabel('t'); 
ylabel('Capital Utilization');

%(d) Calculate the standard deviation of simulated n, u, and inv′. Explain what you would
% qualitatively expect would occur to the standard deviation of each in the following cases.

std_n = std(simulate_n(drop+1:T));
std_u = std(simulate_u(drop+1:T));
std_inv = std(simulate_inv(drop+1:T));

% (a) when theta_2 doubled.
% When this parameter is higher, this causes depreciation to rise more
% quickly. As such, firms experience faster depreciation when using
% capital. Firms experience more volitility in capital utilization so i 
% expect u standard deviation to rise.
% Firms will then invest more to offset these depreciations, increasing
% invesatmnt standard deviation. volitile capital utilization may lead to
% more volitile income, increasing volitility of labor supply.


%(b) The real interest rate doubled.
% Higher interest rate increases return on investment but may also
% introduce more riskiness so I expect standard deviation of investment to
% rise. Higher investment may increase capital utilization, allowing for
% smoother utilkization so i expect stabndard deviation of u to decrease.
% Higher capital utilization may increase output, makeing wage less
% volitile, decreasing standrad deviation of labor supply.