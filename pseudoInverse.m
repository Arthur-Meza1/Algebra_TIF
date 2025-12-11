function A_plus = pseudoInverse (A, tol)
% Pseudoinversa de Moore-Penrose usando SVD
% Entrada: A matriz (mxn)
%          tol tolerancia (opcional)
% Salida: A_plus pseudoinversa (nam)

if nargin < 2
    tol = max(size(A)) * eps(norm(A));
end

% Calcular SVD
[U, S, V] = svd(A, 'econ');
s = diag(S);

% Valores singulares significativos
r = sum(s > tol);

if r == 0
    A_plus = zeros(size(A'));
else
    S_inv = diag(1 ./ s(1:r));
    A_plus = V(:, 1:r) * S_inv * U(:, 1:r)';
end
end