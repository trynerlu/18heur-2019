% simulated annealing

clear

Xsa = load('Xsa2.mat');
Xsa = Xsa.Xsa;

%% Obecne parametry pro vsechny metody
[n m] = size(Xsa);

% number of random shootings
numRS = 1000;

% pocet skoku do okolniho lokalniho minima
skoky = 9;

% MAXKROK =   [1000 500 250 200 125 100 50 40 25 20 10 8 5 4 2 1];
% HX =        [1 2 4 5 8 10 20 25 40 50 100 125 200 250 500 1000];
MAXKROK = 250;
HX = 4;

% neighbor coef - jak daleko koukam do stran
k = 8;

% snizovani teploty
deltaT = 0.9;
% deltaT = [0.99 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2];
% deltaT = [0.1 0.05 0.001];

% FSA
mutation = 'Cauchy';

alpha = 1;

r = 0.01;

for I = 1:length(deltaT)
    for J = 1:length(MAXKROK)
        
        maxkrok = MAXKROK(J);
        
        H = HX(J);
        
        % volba metod
        % 2D vektor vs 3D plocha -> '2D' vs '3D'
        % minima pro kazdy radek nebo jen pro plochu -> 'vector' vs 'normal'
        dim = '3D';
        method = 'normal';
        
        %% SA parametry
        % initial temperature
        T = 100;
        
        % preddefinovani vektoru pro vysledky
        if dim == '3D'
            C = zeros(1,numRS);
        else
            C = zeros(1,length(Xsa));
        end
        
        %% random shooting
        
        if n == 1
            x0 = randi(m);
        elseif m == 1
            y0 = randi(n);
            Xsa = Xsa';
            [n m] = size(Xsa);
        else
            if dim == '3D'
                % kolikrat vystrelime na zacatku
                Y0 = randi(n,1,numRS);
                X0 = randi(m,1,numRS);
            else
                % random shooting pro kazdy row
                X0 = randi(m,1,n);
            end
        end
        
        % current (initial) state/states
        if dim == '3D'
            for i = 1:numRS
                C(1,i) = Xsa(Y0(1,i),X0(1,i));
            end
            
        else
            for i = 1:n
                C(1,i) = Xsa(i,X0(1,i));
                D(1,i) = Xsa(i,X0(1,i));
                
                if method == 'vector'
                    % lokalni maxima pro kazdy row
                    o = (Xsa(i,:) == min(Xsa(i,:)));
                    [row,col] = find(o~=0);
                end
            end
            
        end
        
        %% Random Shooting parametry
        % stejna pocatecni pozice jak SA
        D = C;
        Dmean = mean(D);
        
        
%         %% Shoot and Go / Random Restart Hill Climbing
%         % stejna pocatecni pozice jak SA
%         F = C;
%         
%         Fmean = zeros(1,skoky);
%         
%         X0sg = X0;
%         Y0sg = Y0;
%         
%         SGtime = tic;
%         for poskok = 1:skoky
%             Fmean(poskok) = mean(F);
%             for j = 1:numRS
%                 % podle k
%                	[~,~,xmin,ymin] = soused(Xsa,X0sg(j),Y0sg(j),k);
%                 F(j) = Xsa(ymin,xmin);
%                 X0sg(j) = xmin;
%                 Y0sg(j) = ymin;
%             end
%         end
%         SGtime = toc(SGtime);
        
        %% Simulated Annealing Algoritmhs
        krok = -1;
        kroky = 0;
        Cmean = zeros(1,skoky);
        
        SAtime = tic;
        while krok <= maxkrok
            krok = krok +1;
            
            % -----------------------------------------------------------------
%                 %% ADVANCED
%             
%                 % pres vsechny body
%                 for j = 1:numRS
%             
%                     % hodnoty puvodniho bodu
%                     NEW(1,1) = C(j);        % hodnota
%                     NEW(2,1) = X0(j);       % pozice x
%                     NEW(3,1) = Y0(j);       % pozice y
%             
%                     % H novych bodu
%                     for i = 1:H
%                         kroky = kroky +1;
%                         % vygeneruju novy bod
%                         % lokalni SA
% %                         [xnew,ynew,~,~] = soused(Xsa,X0(j),Y0(j),k);
%                         % globalni SA
%                         xnew = randi(m);
%                         ynew = randi(n);
%             
%                         % postupne nageneruju vsechny novy stavy
%                         NEW(1,i+1) = Xsa(ynew,xnew);
%                         NEW(2,i+1) = xnew;
%                         NEW(3,i+1) = ynew;
%                     end
%             
%                     % ted najdu nejlepsi hodnotu z N
%                     NEWmin = min(NEW(1,:));
%                     [row col] = find(NEW(1,:) == NEWmin);
%             
%                     if length(col) ~= 1
%                         kroky = kroky +1;
%                         r = randi(length(col));
%                         col = col(r);
%                         % nejlepsi stav a jeho pozice
%                         NEWbest = NEW(1,col);
%                         xbest = NEW(2,col);
%                         ybest = NEW(3,col);
%                     else
%                         kroky = kroky +1;
%                         % nejlepsi stav a jeho pozice
%                         NEWbest = NEW(1,col);
%                         xbest = NEW(2,col);
%                         ybest = NEW(3,col);
%                     end
%             
%                     % energy
%                     E = NEWbest-C(j);
%                     % prob
%                     P = 1/(1+exp(-E/T));
%                     r = rand;
%                     if r > P
%                         C(j) = NEWbest;
%                         X0(j) = xbest;
%                         Y0(j) = ybest;
%                     end
%             
%             
%                 end
%             
%                 T = deltaT(I)*T;
            
            %     -----------------------------------------------------------------
            %% NORMAL
%             for i = 1:H
%                 kroky = kroky +1;
%                 %         Cmean(kroky) = mean(C);
%                 %% neighbor choose
%                 if dim == '3D'
%                     
%                     for j = 1:numRS
%                         % podle k
% %                         [xnew,ynew,~,~] = soused(Xsa,X0(j),Y0(j),k);
%                         % globalni SA
%                         xnew = randi(m);
%                         ynew = randi(n);
%                         
%                         % new state
%                         new = Xsa(ynew,xnew);
%                         % energy
%                         E = new-C(j);
%                         % prob
%                         P = 1/(1+exp(-E/T));
%                         r = rand;
%                         
%                         if r > P
%                             C(j) = new;
%                             X0(j) = xnew;
%                             Y0(j) = ynew;
%                         end
%                     end
%                 else
%                     for j = 1:n
%                         [xnew,ynew] = soused(Xsa(j,:),X0(j),j,k);
%                         new = Xsa(ynew,xnew);
%                         E = new-C(j);
%                         P = 1/(1+exp(-E/T));
%                         r = rand;
%                         if r > P
%                             C(j) = new;
%                             X0(j) = xnew;
%                         end
%                     end
%                 end
%                 
%             end
%             
%             % cooling T
%             T = deltaT(I)*T;
            
            %% Fast Simulated Annealing
            %
            % alpha ... cooling strategy par 
            % n0 = pocet vystrelu = numRS
            % k ... pocet kroku = krok
            %
            %
            FSAtime = tic;
            for i = 1:H
                kroky = kroky +1;
                for j = 1:numRS
                    
                    % mutation
                    if mutation == 'normal'
                        % normalni
                        xnew = randi(m);
                        ynew = randi(n);   
                        
                    else
                        % Cauchy
                        u1 = rand;
                        u2 = rand;
                        
                        % r si volim
                        xnew = round(X0(j) + r*tan(pi*(u1-1/2)));
                        ynew = round(Y0(j) + r*tan(pi*(u2-1/2)));
                        
                        % korekce pokud prejedeme pres hranu
                        % periodic extention
                        while xnew > m
                            xnew = xnew - m;
                        end
                        while xnew < 1 
                            xnew = xnew + m;
                        end
                        while ynew > n
                            ynew = ynew - n;
                        end
                        while ynew < 1 
                            ynew = ynew + n;
                        end
                    end
                    
                    % new state
                    new = Xsa(ynew,xnew);
                        
                    s = (C(j)-new)/T;
                    
                    swap = rand;
                    vypocet = 1/2 + atan(s)/pi;
                    
                    if swap < vypocet
                        C(j) = new;
                    	X0(j) = xnew;
                      	Y0(j) = ynew;
                    end
                    
                end
            end
            
            if alpha > 0
                T = T/(1+((krok+1)/numRS)^(alpha));
            else
                T = T*exp(-((krok+1)/numRS)^(-alpha));
            end
            
            
            
            
        end
        SAtime = toc(SAtime);
%         FSAtime = toc(FSAtime);
        disp('========================================')
        disp(['n = ',num2str(maxkrok)])
        disp(['k = ',num2str(H)])
%         disp(['deltaT = ',num2str(deltaT(I))])
%         disp(['alpha = ',num2str(alpha)])
%         disp(['r = ',num2str(r)])
        % disp(['Dostatecna teplota SA = ',num2str(T)])
        % disp(['Pocet kroku snizeni teploty SA = ',num2str(krok)])
        % disp('--------------------')
        disp(['Prumerna energie SA = ',num2str(mean(C))])
        % disp(['Minimalni energie SA = ',num2str(min(C))])
        % disp('--------------------')
        % disp(['Prumerna energie RS = ',num2str(mean(D))])
        % disp(['Minimalni energie RS = ',num2str(min(D))])
        % disp('--------------------')
%         disp(['Prumerna energie SG = ',num2str(mean(F))])
        % disp(['Minimalni energie SG = ',num2str(min(F))])
        % disp('--------------------')
        % disp(['Minimalni energie funkce = ',num2str(min(min(Xsa)))])
        % disp('--------------------')
        disp(['Pocet presnych zasahu SA = ', num2str(num2str(sum(C==min(min(Xsa)))))])
        % disp(['Pocet presnych zasahu RS = ', num2str(num2str(sum(D==min(min(Xsa)))))])
%         disp(['Pocet presnych zasahu SG = ', num2str(num2str(sum(F==min(min(Xsa)))))])
        % disp('--------------------')
%         disp(['Doba trvani SG = ',num2str(SGtime)])
        disp(['Doba trvani SA = ',num2str(SAtime)])
        % disp(['Pocet kroku SA = ',num2str(krok-1)])
        
    end
end