function leer_nombres_imagenes(carpeta::AbstractString)
    nombres = String[]
    for archivo in readdir(carpeta)
        if isfile(joinpath(carpeta, archivo))
            archivo = replace(archivo, ".png" => "")
            push!(nombres, archivo)
        end

    end
    #impirmir numero de imagenes
    println("Numero de imagenes: ", length(nombres))

    return nombres
end

carpeta = "dataset/train/Moderate/" # Reemplaza "/ruta/a/la/carpeta" con la ruta de tu carpeta de imágenes
nombres_imagenes = leer_nombres_imagenes(carpeta)

coordenadas = Tuple{Float64,Float64}[]
for nombre in nombres_imagenes
    coord_y, coord_x = split(nombre, "_")[3:4]
    push!(coordenadas, (parse(Float64, coord_x), parse(Float64, coord_y)))
end


output_file = "./coordenadasModerate.txt"
using DelimitedFiles

coordenadas_str = string.(coordenadas)
coordenadas_str = join(coordenadas_str, ",\n")
coordenadas_str = "[" * coordenadas_str * "]"

open(output_file, "w") do file
    write(file, coordenadas_str)
end



