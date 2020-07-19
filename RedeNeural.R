


#-------------------- REDE Neural / Deep learning / 2 Hidden Layers

# Definindo funções 
# Função Sigmoid
#--------------------------------

sig = function(soma) {
  return (1 / (1 + exp(-soma)))
}

# Função derivada da sigmoid
#--------------------------------

derSig = function(sig) {
  return (sig * (1 - sig))
}

# Criando as variáveis de entrada, matrizes e pesos das sinapses
#-----------------------------------------------------------

In = matrix(c(0,0,1,0,
              0,0,1,1,
              0,1,0,0,
              0,1,0,1,
              0,1,1,0,
              0,1,1,1,
              1,0,0,0,
              1,0,0,1,
              1,0,1,0,
              1,0,1,1,
              1,1,0,0,
              1,1,0,1,
              1,1,1,0,
              1,1,1,1), nrow = 14, ncol = 4, byrow = T)
Out_to_Learn = matrix(c(0,0,1,1,0,0,1,1,0,0,1,1,0,1))
w_In = matrix(runif(14, min = -1, max = 1), nrow = 4, ncol = 4, byrow = T)
w_h0 = matrix(runif(14, min = -1, max = 1), nrow = 4, ncol = 4, byrow = T)
w_h1 = matrix(runif(4, min = -1, max = 1), nrow = 4, ncol = 1, byrow = T)

epocas = 10000
moment = 1
learn_tax = 0.3
m_err = 1
ctdor = 0

# Inicia o processo de aprendizagem
#-------------------------------------------------------------------

while(m_err > 0.005){
  ctdor <- ctdor+1
  
  Cam_In = In
  soma_Cam_In = Cam_In %*% w_In
  Cam_h0  = sig(soma_Cam_In)
  
  Cam_h0 = Cam_h0
  soma_Cam_h0 = Cam_h0 %*% w_h0
  Cam_h1  = sig(soma_Cam_h0)
  
  soma_Cam_h1 = Cam_h1  %*% w_h1
  Out = sig(soma_Cam_h1)
  
# --- Aqui comecam as correcoes 
#-------------------------------------------------
  # Cálculo da Média Absoluta do Erro
  #----------------------------------
  err_Out = Out_to_Learn - Out
  err_Abs = mean(abs(err_Out))
  # print(paste('epoca: ', j))
  print(paste('Erro: ', err_Abs))
  m_err = err_Abs
  
  # Calcula a derivada da sigmoid da saída (backpropagation)
  #_____________________________________
  
  deriv_Out = derSig(Out)
  delt_Out = err_Out * deriv_Out
  
  T_w_h1 = t(w_h1)
  delt_Out_w_h1 = delt_Out %*% T_w_h1
  delt_C_h1 = delt_Out_w_h1 * derSig(Cam_h1)
  
  Cam_h1_T = t(Cam_h1)
  w_h1_new = Cam_h1_T %*% delt_Out
  w_h1 = (w_h1 * moment) + (w_h1_new * learn_tax)
  
  T_w_h0 = t(w_h0)
  delt_Out_w_h0 = delt_C_h1 %*% T_w_h0
  delt_C_h0 = delt_Out_w_h0 * derSig(Cam_h0)
  
  Cam_h0_T = t(Cam_h0)
  w_h0_New = Cam_h0_T %*% delt_C_h1
  w_h0 = (w_h0 * moment) + (w_h0_New * learn_tax)
  
  T_w_in = t(w_In)
  delt_Out_w_in = delt_C_h0 %*% T_w_in
  delt_C_in = delt_Out_w_in * derSig(Cam_In)
  
  Cam_In_T = t(Cam_In)
  w_In_New = Cam_In_T %*% delt_C_in
  w_In = (w_In * moment) + (w_In_New * learn_tax)
  
  }
Out = round(Out, 0)
print(Out)
print(ctdor)


# Teste
# Calcula a saída, baseando-se nas duas entradas de 4 bits iniciais que não foram fornecidas para aprendizado.

# Aqui só se faz necessária a função sigmóide, vez que a função derivada é utilizada para correção dos pesos,
# o que não é mais necessário.

sig = function(soma) {
  return (1 / (1 + exp(-soma)))
}

# Aqui, forneço as duas entradas novas e desconhecidas pela Rede Neural
#_____________________________________________________
In = matrix(c(0,0,0,0,
              0,0,0,1), nrow = 2, ncol = 4, byrow = T)

# Nesta fase a Rede Neural começa a calcular a saída com base nos dados de entrada, fazendo as devidas operações
# de somatória da multiplicação dos valores das camadas vezes os respectivos pesos, e aplicando-os nas funções de
# ativação de cada camada, até camada final.
#________________________________________________
Cam_In = In
soma_Cam_In = Cam_In %*% w_In
Cam_h0  = sig(soma_Cam_In)
Cam_h0 = Cam_h0
soma_Cam_h0 = Cam_h0 %*% w_h0
Cam_h1  = sig(soma_Cam_h0)
soma_Cam_h1 = Cam_h1  %*% w_h1
Out = sig(soma_Cam_h1)

# Agora será printada a saída que está na variável Out sem arredondamento e com arredondamento

print(Out)

Out = round(Out, 0)

print(Out)
