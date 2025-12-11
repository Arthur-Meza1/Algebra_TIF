function [Q, R] = householderQR(A)
% Descomposicion QR mediante transformaciones de Householder
[m, n] = size(A);
R = A;
Q = eye(m);
for k = 1:n
    % Vector a transformar
    x = R(k:m, k);
    norm_x = norm(x);

    if norm_x < eps
        continue;
    end

    % Signo para estabilidad numerica
    if x(1) >= 0
        sigma = norm_x;
    else
        sigma = -norm_x;
    end
    
    % Vector de Householder
    v = x;
    v(1) = v(1) + sigma;
    beta = 2 / (v' * v);
    
    % Actualizar R
    R(k:m, k:n) = R(k:m, k:n) - beta * v * (v' * R(k:m, k:n));
    
    % Actualizar Q
    Q(:, k:m) = Q(:, k:m) - beta * (Q(:, k:m) * v) * v';
end
end