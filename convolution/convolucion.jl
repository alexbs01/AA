module conv

	using Flux
	using Flux.Losses
	using Flux: onehotbatch, onecold, adjust!
	using JLD2, FileIO
	using Statistics: mean


	include("../fonts/boletin04.jl")
	using .Metrics: confusionMatrix, printConfusionMatrix

	function convExc(topology::AbstractArray{<:Int, 1}, filter::Int, num::Int)
		

		file = "comv.jld2"


		in = load(file, "im")
		tr = load(file, "tag")

		tra = Int32.(trunc(size(in, 1) * 0.7))
		val = tra + Int32.(trunc(size(in, 1) * 0.2))

		train_imgs = in[1:tra, :]
		train_labels = tr[1:tra, :]
		validation_imgs = in[tra+1:val, :]
		validation_labels = tr[tra+1:val, :]
		test_imgs = in[val+1:end, :]
		test_labels = tr[val+1:end, :]

		labels = [0; 1; 2]; # Las etiquetas

		in = nothing;
		tr = nothing;
		GC.gc()

		function convertirArrayImagenesHWCN(imagenes)
			numPatrones = length(imagenes)
			nuevoArray = Array{Float32, 4}(undef, 64, 64, 3, numPatrones) # Importante que sea un array de Float32
			for i in 1:numPatrones
				@assert (size(imagenes[i]) == (64, 64, 3)) "Las imagenes no tienen tamaño 320x320"
				nuevoArray[:, :, 1, i] .= imagenes[i][:, :, 1]
				nuevoArray[:, :, 2, i] .= imagenes[i][:, :, 2]
				nuevoArray[:, :, 3, i] .= imagenes[i][:, :, 3]
			end
			return nuevoArray
		end;

		train_imgs = convertirArrayImagenesHWCN(train_imgs);
		validation_imgs = convertirArrayImagenesHWCN(validation_imgs);
		test_imgs = convertirArrayImagenesHWCN(test_imgs);


		println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
		println("Tamaño de la matriz de validación: ", size(validation_imgs))
		println("Tamaño de la matriz de test:          ", size(test_imgs))


		# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
		# En otro caso, habria que normalizarlas
		println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");


		batch_size = 128
		# Creamos los indices: partimos el vector 1:N en grupos de batch_size
		gruposIndicesBatch = Iterators.partition(1:size(train_imgs, 4), batch_size);
		println("Se han creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


		train_set = [(train_imgs[:, :, :, indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

		validation_set = (validation_imgs, onehotbatch(validation_labels, labels));

		# Creamos un batch similar, pero con todas las imagenes de test
		test_set = (test_imgs, onehotbatch(test_labels, labels));

		# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
		train_imgs = nothing;
		test_imgs = nothing;
		GC.gc(); # Pasar el recolector de basura

		funcionTransferenciaCapasConvolucionales = relu;

		# Definimos la red con la funcion Chain, que concatena distintas capas

		#=
		ann = Chain(
			Conv((2, 2), 3 => 16, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),
			Conv((2, 2), 16 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),
			Conv((2, 2), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),
			x -> reshape(x, :, size(x, 4)),
			Dense(2048, 5),	
			Dense(5, 4),
			Dense(4, 3),
			softmax
		)
		=#

		ann = buildANN(filter, topology, 3, num)

		numBatchCoger = 1;
		numImagenEnEseBatch = [12, 6];

		entradaCapa = train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch];

		ann(train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch]);



		# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
		loss(ann, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);
		# Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
		accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
		# Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


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
			#global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin

			# Se entrena un ciclo
			Flux.train!(loss, ann, train_set, opt_state)

			numCiclo += 1

			# Se calcula la precision en el conjunto de entrenamiento:
			precisionEntrenamiento = mean(accuracy(validation_set))
			println("Ciclo ", numCiclo, ": Precision en el conjunto de validación: ", 100 * precisionEntrenamiento, " %")

			# Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
			if (precisionEntrenamiento > mejorPrecision)
				mejorPrecision = precisionEntrenamiento
				precisionTest = accuracy(test_set)
				println("   Mejora en el conjunto de validación -> Precision en el conjunto de test: ", 100 * precisionTest, " %")
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

		printConfusionMatrix(onecold(ann(test_set[1]), [0; 1; 2]), vec(onecold(test_set[2], [0; 1; 2])))

	end

	function buildANN(filter::Int, topology::AbstractArray{<:Int, 1}, numOutputs::Int, inputs::Int)

		funcionTransferenciaCapasConvolucionales = relu;

		ann = Chain(
			Conv((filter, filter), 3 => 16, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),			
			Conv((filter, filter), 16 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),			
			Conv((filter, filter), 32 => 32, pad = (1, 1), funcionTransferenciaCapasConvolucionales),
			MaxPool((2, 2)),			
			x -> reshape(x, :, size(x, 4)),
		)
		numInputsLayer = inputs
		iterTransFunction = 1

		if length(topology) != 0 && topology[1]!=0
			for numOutputsLayer ∈ topology
				ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer))
				numInputsLayer = numOutputsLayer
				iterTransFunction += 1
			end
		end

		ann = Chain(ann..., Dense(numInputsLayer, numOutputs))
		ann = Chain(ann..., softmax)

		println(ann)

		return ann
	end

end