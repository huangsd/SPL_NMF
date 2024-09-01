% min ||X-FG^T||_{2,1}
% s.t. G^T*G=I
%  prestd(p);                                           
function G_res = SPLNMF(X, G, F, labels, maxiter, is_soft)
% X: mfea * nSmp  
% G: nSmp * k
% F: mfea * k
% is_soft: use the soft weight or not. 1: soft weight, 0: hard weight.
disp('SPL-NMF');
% % =====================   Normalization =====================
[mfea, nSmp] = size(X);
% for  j = 1:nSmp
%      X(j,:) = ( X(j,:) - mean( X(j,:) ) ) / std( X(j,:) ) ;
% end
% for i = 1:nSmp
%    % X(:,i)=X(:,i)./sum(X(:,i));
%    % normalize each data point to unit length
%      X(:,i)=X(:,i)/(sqrt(sum(X(:,i).*X(:,i), 1))+eps); 
%    % X(:,i)=X(:,i)/(sqrt(X(:,i)'*X(:,i))+eps);
% end
X = mapminmax(X,0,1);
%X = X';
D = eye(nSmp, nSmp);

% Gres = kmeans(X',length(unique(labels)),'emptyaction','singleton');
% G = zeros(nSmp,length(unique(labels)));
% for i = 1:nSmp
%     G(i,Gres(i)) = 1;
% end
% G = G+0.2;
% 
% Fres = kmeans(X,length(unique(labels)),'emptyaction','singleton');
% F = zeros(mfea,length(unique(labels)));
% for i = 1:mfea
%     F(i,Fres(i)) = 1;
% end
% F = F+0.2;
ChosenNum = ceil(0.6*nSmp);
stopc = 0; % stop condition
for i = 1:10

    for iter = 1:maxiter
        
         F = F.*sqrt((X*D*G)./(F*G'*D*G+eps));
  
         G = G.*sqrt((D*X'*F)./(D*G*G'*X'*F+eps));

         Delta = X - F*G';
         DeltaN = sqrt(sum(Delta.*Delta, 1));
    
         if iter == 1
            DeltaS = sort(DeltaN);
            lambda = DeltaS(ChosenNum);
            zeta = lambda*0.5;  % set you own thresholding parameter
         end   
%          for j = 1:nSmp
%              DeltaN(j) = DeltaN(j)*Weight(DeltaN(j),lambda,zeta);
%          end
         weight = zeros(1,nSmp);
         if is_soft == 1
             for j = 1:nSmp
                  weight(j) = SoftWeight(DeltaN(j),lambda,zeta); % soft weight
             end
         else
             for j = 1:nSmp
                  weight(j) = HardWeight(DeltaN(j),lambda); % hard weight
             end
         end
             
          DiagD = 0.5./(DeltaN+eps);
          DiagD = DiagD.*weight;
          D = diag(DiagD);
                      
    %--------obj----------
    % compute the obj value after all the instances are chosen
          thresh = 1e-6;
          if stopc == 1
             T = X - F*G';
             obj = sum(diag(T*D*T'));
             myobj(iter) = obj;
             if iter > 2 
                diff = abs(myobj(iter-1) - myobj(iter));
                if(diff < thresh)
                   break;
                end
             end 
          end
             
    end   
    %fprintf('%d-th iteration, obj = %f\n', iter, obj);
    
    if stopc == 1 
       break;
    end
    
    ChosenNum = ChosenNum + ceil(nSmp/10); % increase ChosenNum in steps of 10%
    if ChosenNum > nSmp
       ChosenNum = nSmp; % all the instances are chosen
       stopc = 1; % stopc = 1
    end
    
end
fprintf('SPLNMF--SPL iterations:%d\n', i);
fprintf('total iterations:%d\n', iter);
% G_res = zeros(nSmp,1);
% for j = 1:nSmp
%     [tmp G_res(j)] = max(G(j,:));
% end
 G_res = kmeans(G,length(unique(labels)),'emptyaction','singleton');

end



function weight = SoftWeight(x,lambda,zeta)
    %
    % compute the soft weight for x
    % lambda:  thresholding parameter
    % zeta: parameter
    % zeta < lambda
    % zeta is uauslly set as 0.5*lambda, i.e., zeta = 0.5*lambda
    % Written by Shudong Huang  huangsd@std.uestc.edu.cn
    % 2017-09-25
    % 
    if zeta > lambda
        zeta = 0.5*lambda;
    end
    
    if x >= lambda
        weight = 0;
    else if x <= (zeta*lambda)/(zeta+lambda)
            weight = 1;
        else
            weight = zeta*((lambda-x)/(x*lambda+eps));
        end
    end
    
end

function weight = HardWeight(x,lambda)
%
% hard weight
%
  if x <= lambda
      weight = 1;
  else
      weight = 0;
  end

end





