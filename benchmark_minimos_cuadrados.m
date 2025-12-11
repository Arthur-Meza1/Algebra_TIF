
clear all; close all; clc;
m = 200; 
n = 50;  
cond_numbers = logspace(1, 10, 10);
n_trials = length(cond_numbers);

errors_ne = zeros(n_trials, 1);
errors_qr = zeros(n_trials, 1);
errors_svd = zeros(n_trials, 1);
times_ne = zeros(n_trials, 1);
times_qr = zeros(n_trials, 1);
times_svd = zeros(n_trials, 1);

for i = 1:n_trials
    cond_num = cond_numbers(i);
    
    % Generar matriz A con numero de condicion especifico
    A = gallery('randsvd', [m, n], cond_num);
    
    % Generar la solucion verdadera y el vector b con ruido
    x_true = randn(n, 1);
    b = A * x_true + 0.01 * randn(m, 1); % Añadir ruido a la medicion
    
    % -----------------------------------------------------------------
    % Metodo 1: Ecuaciones Normales
    % Formula: (A'A)x = A'b -> x = (A'A)\(A'b)
    % -----------------------------------------------------------------
    tic;
    x_ne = (A' * A) \ (A' * b);
    times_ne(i) = toc;
    errors_ne(i) = norm(x_true - x_ne) / norm(x_true);
    
    % -----------------------------------------------------------------
    % Metodo 2: QR (Requiere la funcion householderQR.m)
    % Formula: Rx = Q'b
    % -----------------------------------------------------------------
    tic;
    [Q, R] = householderQR(A);
    R1 = R(1:n, 1:n);           % Extraer la parte nxn triangular
    c = Q(:, 1:n)' * b;         % Calcular c = Q'b
    x_qr = R1 \ c;              % Resolver el sistema triangular superior
    times_qr(i) = toc;
    errors_qr(i) = norm(x_true - x_qr) / norm(x_true);
    
    % -----------------------------------------------------------------
    % Metodo 3: SVD (Pseudoinversa Tikhonov regularizada)
    % Formula: x = V * inv(Sigma_truncada) * U' * b
    % -----------------------------------------------------------------
    tic;
    [U, S, V] = svd(A, 'econ'); 
    s = diag(S);
    
    % Tolerancia para pseudoinversa (ignorar valores singulares cercanos a cero)
    tol = max(m, n) * eps(max(s));
    s_inv = zeros(size(s));
    s_inv(s > tol) = 1 ./ s(s > tol); % Solo invertir valores > tol
    
    x_svd = V * diag(s_inv) * U' * b;
    times_svd(i) = toc;
    errors_svd(i) = norm(x_true - x_svd) / norm(x_true);
end

figure('Name', 'Comparacion de Metodos de Minimos Cuadrados', 'Position', [100, 100, 1200, 500]);

% Grafico 1: Estabilidad (Error vs Condicionamiento)
subplot(1, 2, 1);
loglog(cond_numbers, errors_ne, 'r-o', 'LineWidth', 2, 'DisplayName', 'Ecuaciones Normales');
hold on;
loglog(cond_numbers, errors_qr, 'b-s', 'LineWidth', 2, 'DisplayName', 'QR Householder');
loglog(cond_numbers, errors_svd, 'g-^', 'LineWidth', 2, 'DisplayName', 'SVD');
loglog(cond_numbers, ones(size(cond_numbers)) * eps, 'k--', 'DisplayName', 'Precision Maquina (eps)'); 
xlabel('Numero de Condicion de A (\kappa(A))');
ylabel('Error Relativo (||x_{true} - x_{calc}|| / ||x_{true}||)');
title('Estabilidad: Error vs Condicionamiento de la Matriz');
legend('Location', 'northwest');
grid on;
set(gca, 'FontSize', 10);


% Grafico 2: Eficiencia (Tiempo vs Condicionamiento)
subplot(1, 2, 2);
semilogx(cond_numbers, times_ne, 'r-o', 'LineWidth', 2, 'DisplayName', 'Ecuaciones Normales');
hold on;
semilogx(cond_numbers, times_qr, 'b-s', 'LineWidth', 2, 'DisplayName', 'QR Householder');
semilogx(cond_numbers, times_svd, 'g-^', 'LineWidth', 2, 'DisplayName', 'SVD');
xlabel('Numero de Condicion de A (\kappa(A))');
ylabel('Tiempo de Ejecucion (s)');
title('Eficiencia: Tiempo vs Condicionamiento');
legend('Location', 'northwest');
grid on;
set(gca, 'FontSize', 10);


disp('Simulacion de Benchmark finalizada. Se han generado dos graficos de comparacion.');