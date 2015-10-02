speech = wavread('speech.wav');
BLOCK_LENGTH = 160;
ORDER = 10;
numblocks = ceil(length(speech)/BLOCK_LENGTH);

%% Direct Quantization + MSE
for r = 1:8
    L = 2^r;
    q = (max(speech) - min(speech))/L;
    yq = round(speech/q)*q;
    mse = (speech - yq)'*(speech - yq)/length(speech);
end

% Zero Padding Speech Signal
speech = [speech;zeros(BLOCK_LENGTH*numblocks - length(speech),1)];

%% Parameter Estimation
y = zeros(160,1);
for ii = 0:numblocks-1 
    if ii == 0
        A = toeplitz([0;speech(1:BLOCK_LENGTH-1)],zeros(ORDER,1));
        param = A\speech(1:BLOCK_LENGTH);
        residual = speech(1:BLOCK_LENGTH) - A*param;
        for kk = 1:160
            if kk <= 10
                y(kk) = param'*[flipud(y(1:kk-1));zeros(10-kk+1,1)] + residual(kk);
            else
                y(kk) = param'*flipud(y(kk-10:kk-1)) + residual(kk);
            end
        
        end
    else 
        prevSeg = speech((ii-1)*BLOCK_LENGTH + 1:ii*BLOCK_LENGTH);
        speechSeg = speech(ii*BLOCK_LENGTH + 1:(ii + 1)*BLOCK_LENGTH);
        A = toeplitz([prevSeg(end);speechSeg(1:159)],flipud(prevSeg(end-ORDER+1:end)));
        param = A\speechSeg;
        residual = speechSeg - A*param;
        
        % Manual Residual Calculation + MSE
        error = speechSeg - (param'*toeplitz(flipud(prevSeg(end-ORDER+1:end)),[prevSeg(end);speechSeg(1:159)]))';
        mean((residual - error).^2);
        
        % Reconstruction
        yblock = zeros(160,1);
        for jj = 1:160
            if jj <= 10
                yblock(jj) = param'*[flipud(yblock(1:jj-1));flipud(y(end-10+jj:end))] + residual(jj);
            else
                yblock(jj) = param'*flipud(yblock(jj-10:jj-1)) + residual(jj);
            end
        end
        y = [y;yblock];
    end
end
