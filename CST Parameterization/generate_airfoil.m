%% Main script to generate airfoils from CSV
clear; clc; close all;

% Parameters 
csv_filename = 'G.csv';  % YOUR CSV FILE 
N = 200;                        % Total number of points (approx)
dz = 0;                         % Trailing edge thickness

% Ask user which row to generate
row_number = input('Enter row number to generate: ');

% Generate the airfoil
coord = CST_airfoil_from_csv(csv_filename, row_number, dz, N);

disp('Done! Airfoil saved as .dat file');

%% ============ FUNCTIONS ============

function [coord] = CST_airfoil_from_csv(csv_filename, row_number, dz, N)
% Description : Create a set of airfoil coordinates using CST parametrization 
%               method from weights stored in a CSV file

% Read weights from CSV file (skip headers)
data = readmatrix(csv_filename, 'NumHeaderLines', 1);

% Check if row_number exists (user enters CSV row number)
if row_number > size(data, 1) + 1
    error('Row %d does not exist. File has %d rows.', row_number, size(data, 1) + 1);
end

% Convert user's row number to MATLAB row number (subtract 1 for header)
matlab_row = row_number - 1;

% Extract weights for the selected row
row_data = data(matlab_row, :);

% Split into lower and upper weights (6 each)
num_weights = 6;
wl = row_data(1:num_weights);
wu = row_data(num_weights+1:end);

% Display the weights being used
fprintf('Lower weights: [%s]\n', num2str(wl));
fprintf('Upper weights: [%s]\n', num2str(wu));

% Call original CST function with the extracted weights
coord = CST_airfoil(wl, wu, dz, N, row_number);

end

function [coord] = CST_airfoil(wl, wu, dz, N, row_number)
% Description : Create a set of airfoil coordinates using CST parametrization method 

% Create x coordinate (Cosine spacing clustered at LE and TE)
% Use N/2 points per side to achieve the total N requested
N_half = floor(N/2);
theta = linspace(0, pi, N_half + 1)';
x_base = 0.5 * (1 - cos(theta)); % x_base goes strictly from 0 to 1

% N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
N1 = 0.5;
N2 = 1;

% Call ClassShape function to determine upper and lower surface y-coordinates
[yl] = ClassShape(wl, x_base, N1, N2, -dz); 
[yu] = ClassShape(wu, x_base, N1, N2, dz);  

% ===== Assemble in strict Selig Format =====
% Upper surface goes from Trailing Edge (1) to Leading Edge (0)
x_upper = flip(x_base);
y_upper = flip(yu);

% Lower surface goes from Leading Edge (0) to Trailing Edge (1)
% We start at index 2 to avoid writing the (0,0) coordinate twice!
x_lower = x_base(2:end);
y_lower = yl(2:end);

% Combine upper and lower coordinates
x = [x_upper; x_lower];
y = [y_upper; y_lower];
coord = [x y];

% ===== Plot the airfoil =====
figure('Name', sprintf('CST Airfoil - Row %d', row_number), 'NumberTitle', 'off');
plot(coord(:,1), coord(:,2), 'b-', 'LineWidth', 2);
hold on;
% Plot trailing edge points
plot(coord(1,1), coord(1,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Upper TE
plot(coord(end,1), coord(end,2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); % Lower TE
% Plot leading edge point
plot(0, 0, 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
hold off;
grid on;
axis equal;
xlabel('x/c');
ylabel('y/c');
title(sprintf('CST Airfoil (Row %d, %d total points)', row_number, length(x)));
legend('Airfoil surface', 'Trailing edge', 'Leading edge', 'Location', 'best');

% ===== Save data to .dat file =====
filename = sprintf('G%d.dat', row_number);
save_airfoil_dat(filename, coord);
fprintf('Airfoil data saved to: %s\n', filename);

% Display some information
fprintf('Number of points: %d\n', length(x));
fprintf('Trailing edge thickness: %.4f\n', dz);
[~, idx] = max(abs(coord(:,2)));
fprintf('Max thickness: %.4f at x/c = %.4f\n', max(abs(coord(:,2))), coord(idx, 1));

end

function [y] = ClassShape(w,x,N1,N2,dz)

% Class function; taking input of N1 and N2
C = zeros(size(x,1), 1);
for i = 1:size(x,1)
    C(i,1) = x(i)^N1*((1-x(i))^N2);
end

% Shape function; using Bernstein Polynomials
n = size(w,2)-1; % Order of Bernstein polynomials

K = zeros(1, n+1);
for i = 1:n+1
     K(i) = factorial(n)/(factorial(i-1)*(factorial((n)-(i-1))));
end

S = zeros(size(x,1), 1);
for i = 1:size(x,1)
    S(i,1) = 0;
    for j = 1:n+1
        S(i,1) = S(i,1) + w(j)*K(j)*x(i)^(j-1)*((1-x(i))^(n-(j-1)));
    end
end

% Calculate y output
y = zeros(size(x,1), 1);
for i = 1:size(x,1)
   y(i,1) = C(i,1)*S(i,1) + x(i)*dz;
end

end

function save_airfoil_dat(filename, coord)
% Save airfoil coordinates in XFOIL .dat format
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open file: %s', filename);
end

% Write header (airfoil name)
% Note: Using a generic name without spaces is safer for XFOIL
fprintf(fid, 'CST_Airfoil\n');

% Because coord is ALREADY built in Selig format (Upper TE -> LE -> Lower TE),
% we just print it sequentially from top to bottom.
for i = 1:size(coord, 1)
    fprintf(fid, '%.8f    %.8f\n', coord(i,1), coord(i,2));
end

fclose(fid);
end