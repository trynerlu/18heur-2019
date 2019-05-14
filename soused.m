function [xnew,ynew,xmin,ymin] = soused(Xsa,x0,y0,k)

[n m] = size(Xsa);


%% 1D
% pokud to je vektor - sousedstvi je interval
% a zajima me teda jen x0
if n == 1
    % jsem blizko k levemu rohu
    if x0-k < 1
        N = Xsa(1:x0+k);
        l = length(N);
        p = randi(l);
        xnew = p;
    elseif x0+k > m
        N = Xsa(x0-k:m);
        l = length(N);
        p = randi(l);
        xnew = x0-k-1+p;        
    else
        N = Xsa(x0-k:x0+k);
        l = length(N);
        p = randi(l);
        xnew = x0-k-1+p;    
    end
    ynew = y0;
else
%% mam 2D
    % jsem blizko k levemu rohu na ose x
    if x0-k < 1
        % jsem blizko k hornimu rohu na ose y
        % = horni levy roh
        if y0-k < 1
            N = Xsa(1:y0+k,1:x0+k);
            % vzdy budu uvazovat ctvercove sousedstvi
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = p1;
            ynew = p2;
            
            % lokalni minimum pro SG
            Nymin = 1-1;
            Nxmin = 1-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
        
        % jsem blizko k dolnimu rohu na ose y
        % = dolni levy roh
        elseif y0+k > n
            N = Xsa(y0-k:n,1:x0+k);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = p1;             
            ynew = y0-k-1+p2;
            
            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = 1-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
        
        % jsem v pohode na y
        % = levy okraj
        else
            N = Xsa(y0-k:y0+k,1:x0+k);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1); 
            xnew = p1;
            ynew = y0-k-1+p2;
            
            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = 1-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
        end
        
    % jsem blizko k pravemu rohu na ose x    
    elseif x0+k > m
        % jsem blizko k hornimu rohu na ose y
        % = pravy horni roh
        if y0-k < 1
            N = Xsa(1:y0+k,x0-k:m);
            % vzdy budu uvazovat ctvercove sousedstvi
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = x0-k-1+p1;
            ynew = p2;
            
            % lokalni minimum pro SG
            Nymin = 1-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
            
        % jsem blizko k dolnimu rohu na ose y
        % = dolni pravy roh
        elseif y0+k > n
            N = Xsa(y0-k:n,x0-k:m);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = x0-k-1+p1;             
            ynew = y0-k-1+p2;
            
            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
            
        % na ose y v pohode
        % = pravy okraj
        else
            N = Xsa(y0-k:y0+k,x0-k:m);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1); 
            xnew = x0-k-1+p1;
            ynew = y0-k-1+p2;
            
            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
            
        end
    
    % na ose x jsem v pohode
    % jsem blizko k hornimu okraji na ose y
    % = horni okraj
    else
        if y0-k < 1
            N = Xsa(1:y0+k,x0-k:x0+k);
            % vzdy budu uvazovat ctvercove sousedstvi
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = x0-k-1+p1;
            ynew = p2;
            
            % lokalni minimum pro SG
            Nymin = 1-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
        
        % jsem blizko k dolnimu okraji na ose y
        % = dolni okraj
        elseif y0+k > n
            N = Xsa(y0-k:n,x0-k:x0+k);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1);
            xnew = x0-k-1+p1;             
            ynew = y0-k-1+p2;

            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
            
        % vsechno vsude super
        % = vnitrek matice
        else
            N = Xsa(y0-k:y0+k,x0-k:x0+k);
            [l1 l2] = size(N);
            p1 = randi(l2);
            p2 = randi(l1); 
            xnew = x0-k-1+p1;
            ynew = y0-k-1+p2;
            
            % lokalni minimum pro SG
            Nymin = y0-k-1;
            Nxmin = x0-k-1;
            
            Nmin = min(min(N));
            [row col] = find(N == Nmin);
            
            if length(row) ~= 1
                r1 = randi(length(row));
                r2 = randi(length(row));
                xmin = col(r1) + Nxmin;
                ymin = row(r2) + Nymin;
            else
                xmin = col + Nxmin;
                ymin = row + Nymin;
            end
            
        end
    end
end
    
