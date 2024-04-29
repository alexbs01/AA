
#
# Podéis coger este código y modificarlo para realizar vuestra aproximación basada en Deep Learning
#
# Observaciones con respecto a realizar una aproximación adicional utilizando Deep Learning:
#  - En Deep Learning se suele trabajar con bases de datos enormes, con lo que hacer validación cruzada es muy lento. Por este motivo, aunque desde el punto de vista teórico habría que hacerla, no es algo habitual, por lo que no tenéis que hacerla en esta aproximación.
#  - Igualmente, como las bases de datos son muy grandes, los patrones de entrenamiento se suelen dividir en subconjuntos denominados batches. En vuestro caso, como el número de patrones no va a ser tan grande, esto no es necesario, así que utilizar todos los patrones de entrenamiento en un único batch, como se ha venido haciendo en las prácticas hasta el momento.
#  - Por usar bases de datos tan grandes, en Deep Learning no se suele usar conjunto de validación. En lugar de ello, se suelen usar otras técnicas de regularización. En esta aproximación, podéis probar a entrenar las redes sin validación, y si veis que se sobreentrenan, probad a usar validación.
#  - La RNA que os dejo tiene varias capas convolucionales y maxpool. Cada capa convolucionar implementa filtros de un tamaño para pasar a un número distinto de canales. Ya que no vais a hacer validación cruzada, podéis hacer pruebas con distintas arquitecturas (un entrenamiento con cada una), para hacer una tabla comparativa que poner en la memoria.
#  - En redes convolucionales las imágernes de entrada son de tamaño fijo, porque se tiene una neurona de entrada por pixel de la imagen y canal (RGB o escala de grises). Como hemos estado trabajando con ventanas de tamaño variable, lo que se suele hacer es cambiar el tamaño de las ventanas para que todas tengan el mismo tamaño. Esto se puede hacer con la función imresize, tenéis más documentación en https://juliaimages.org/latest/pkgs/transformations/
#  - Si en lugar de usar redes convolucionales para procesado de imagen se usan para procesado de señales, es necesario realizar modiicaciones adicionales al código. Por ejemplo, el tamaño de los filtros de convolución empleado, aquí (3, 3), en lugar de ser bidimensional (para imagenes) debería ser de una dimensión (para señales).
#  - La base de datos de este ejemplo (MNIST) ya tiene un conjunto de parones que debe ser usado para test. En vuestro caso no será así, y como no se va a realizar validación cruzada, para cada arquitectura habrá que realizar un hold out.
#

using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean

include("../fonts/boletin04.jl")
using .Metrics: confusionMatrix

file = "comvL.jld2"

in = load(file, "im")
tr = load(file, "tag")

tra = Int32.(trunc(size(in, 1) * 0.9))

train_imgs   = in[1:tra, :]
train_labels = tr[1:tra, :]
test_imgs    = in[tra+1:end, :]
test_labels  = tr[tra+1:end, :]

labels = [0; 1; 2]; # Las etiquetas

in = nothing;
tr = nothing;
GC.gc()

# Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
#  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
#  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
#  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
#  Es decir, Height x Width x Channels x N
#  En el caso de esta base de datos
#   Height = 28
#   Width = 28
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:
function convertirArrayImagenesHWCN(imagenes)
	numPatrones = length(imagenes)
	nuevoArray = Array{Float32, 4}(undef, 320, 320, 3, numPatrones) # Importante que sea un array de Float32
	for i in 1:numPatrones
		@assert (size(imagenes[i]) == (320, 320, 3)) "Las imagenes no tienen tamaño 320x320"
		nuevoArray[:, :, 1, i] .= imagenes[i][:, :, 1]
		nuevoArray[:, :, 2, i] .= imagenes[i][:, :, 2]
		nuevoArray[:, :, 3, i] .= imagenes[i][:, :, 3]
	end
	return nuevoArray
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);


println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))


# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
# En otro caso, habria que normalizarlas
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");

# Cuando se tienen tantos patrones de entrenamiento (en este caso 60000),
#  generalmente no se entrena pasando todos los patrones y modificando el error
#  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
#  y se van aplicando uno a uno

# Hacemos los indices para las particiones
# Cuantos patrones va a tener cada particion
batch_size = 50
# Creamos los indices: partimos el vector 1:N en grupos de batch_size
gruposIndicesBatch = Iterators.partition(1:size(train_imgs, 4), batch_size);
println("Se han creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
#     onehotbatch(train_labels[indicesBatch], labels)
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
# Sólo resta iterar por cada batch para construir el vector de batches
train_set = [(train_imgs[:, :, :, indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
test_set = (test_imgs, onehotbatch(test_labels, labels));

# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura

funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
ann = Chain(Conv((3, 3), 3 => 16, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), Conv((3, 3), 16 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), Conv((3, 3), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), Conv((3, 3), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), Conv((3, 3), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), Conv((3, 3), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
	MaxPool((2, 2)), x -> reshape(x, :, size(x, 4)),
	Dense(800, 3), softmax)

# Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
# Usaremos como entrada varios patrones de un batch
numBatchCoger = 1;
numImagenEnEseBatch = [12, 6];
# Para coger esos patrones de ese batch:
#  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
#  Por tanto:
#   train_set[numBatchCoger] -> La tupla del batch seleccionado
#   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
#   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
entradaCapa = train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch];
numCapas = length(Flux.params(ann));
println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
	println("   Capa ", numCapa, ": ", ann[numCapa])
	# Le pasamos la entrada a esta capa
	global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
	capa = ann[numCapa]
	salidaCapa = capa(entradaCapa)
	println("      La salida de esta capa tiene dimension ", size(salidaCapa))
	entradaCapa = salidaCapa
end

# Sin embargo, para aplicar un patron no hace falta hacer todo eso.
#  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
ann(train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch]);



# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
loss(ann, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);
# Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
# Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


# Mostramos la precision antes de comenzar el entrenamiento:
#  train_set es un array de batches
#  accuracy recibe como parametro un batch
#  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
#   y devuelve un array con los resultados
#  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
#   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100 * mean(accuracy.(train_set)), " %");


# Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
eta = 0.01;
opt_state = Flux.setup(Adam(eta), ann);


println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while !criterioFin

	# Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
	global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin

	# Se entrena un ciclo
	Flux.train!(loss, ann, train_set, opt_state)

	numCiclo += 1

	# Se calcula la precision en el conjunto de entrenamiento:
	precisionEntrenamiento = mean(accuracy.(train_set))
	println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100 * precisionEntrenamiento, " %")

	# Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
	if (precisionEntrenamiento > mejorPrecision)
		mejorPrecision = precisionEntrenamiento
		precisionTest = accuracy(test_set)
		println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100 * precisionTest, " %")
		mejorModelo = deepcopy(ann)
		numCicloUltimaMejora = numCiclo
	end

	# Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
	if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1.0e-5)
		global eta
		eta /= 10.0
		println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta)
		adjust!(opt_state, eta)
		numCicloUltimaMejora = numCiclo
	end

	# Criterios de parada:

	# Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
	if (precisionEntrenamiento >= 0.999)
		println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
		criterioFin = true
	end

	# Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
	if (numCiclo - numCicloUltimaMejora >= 10)
		println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
		criterioFin = true
	end
end

(
	acc,
	errorRate,
	sensibility,
	specificity,
	precision,
	negativePredictiveValues,
	f1,
	matrix,
) = confusionMatrix(onecold(ann(test_set[1]),[0;1;2]), vec(onecold(test_set[2], [0;1;2])))

println("\nAccuracy: ", acc)
println("Error rate: ", errorRate)
println("Sensitivity: ", sensibility)
println("Specificity: ", specificity)
println("Precision: ", precision)
println("Negative predictive value: ", negativePredictiveValues)
println("F1 score: ", f1)
println(matrix)