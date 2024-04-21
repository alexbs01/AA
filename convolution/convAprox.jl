using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean

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

file = "comvL.jld2"

in = load(file, "im")
tr = load(file, "tag")

tra = Int32.(trunc(size(in, 1) * 0.9))

train_imgs   = in[1:tra, :]
train_labels = tr[1:tra, :]
test_imgs    = in[tra+1:end, :]
test_labels  = tr[1:tra, :]
labels       = 0:9

train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);
println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");

batch_size = 128
gruposIndicesBatch = Iterators.partition(1:size(train_imgs, 4), batch_size);
println("Se han creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");

train_set = [(train_imgs[:, :, :, indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];
test_set = (test_imgs, onehotbatch(test_labels, labels));

train_imgs = nothing;
test_imgs = nothing;
GC.gc()

ann = Chain(
	Conv((3, 3), 3 => 16, pad = (1, 1), relu),
	MaxPool((2, 2)), Conv((3, 3), 16 => 32, pad = (1, 1), relu),
	MaxPool((2, 2)), Conv((3, 3), 32 => 32, pad = (1, 1), relu),
	MaxPool((2, 2)), x -> reshape(x, :, size(x, 4)),
	Dense(51200, 1, σ),
)

numBatchCoger = 1;
numImagenEnEseBatch = [12, 6];

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

ann(train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch]);

loss(ann, x, y) = Losses.mse(ann(x), y);
mae(batch) = mean(abs.(ann(batch[1]) .- batch[2]));

opt_state = Flux.setup(Adam(0.001), ann);


mejorMae = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;
while !criterioFin

	# Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
	global numCicloUltimaMejora, numCiclo, mejorMae, mejorModelo, criterioFin

	# Se entrena un ciclo
	Flux.train!(loss, ann, train_set, opt_state)

	numCiclo += 1

	# Se calcula la precision en el conjunto de entrenamiento:
	maeEntrenamineto = mean(mae.(train_set))
	println("Ciclo ", numCiclo, ": Mae en el conjunto de entrenamiento: ", 100 * maeEntrenamineto, " %")

	# Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
	if (maeEntrenamineto >= mejorMae)
		mejorMae = maeEntrenamineto
		maeTest = mae(test_set)
		println("   Mejora en el conjunto de entrenamiento -> Mae en el conjunto de test: ", 100 * maeTest, " %")
		mejorModelo = deepcopy(ann)
		numCicloUltimaMejora = numCiclo
	end

	# Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
	if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
		opt.eta /= 10.0
		println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta)
		numCicloUltimaMejora = numCiclo
	end

	# Criterios de parada:

	# Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
	if (maeEntrenamineto <= 0.01)
		println("   La Mae ha llegado al 0.01")
		criterioFin = true
	end

	# Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
	if (numCiclo - numCicloUltimaMejora >= 10)
		println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
		criterioFin = true
	end
end


