import Pkg
Pkg.add("Arrow")
Pkg.add("Images")
Pkg.add("DataFrames")

using Arrow, DataFrames, Images

# Ruta al archivo .parquet
ruta_archivo = "dataset/train-00000-of-00024.parquet"

# Leer el archivo .parquet
table = Arrow.Table(ruta_archivo)

# Imprimir la tabla para verificar si se lee correctamente
println("Tabla:")
show(table)

# Verificar las columnas disponibles en la tabla
println("Columnas disponibles en la tabla:")
println(collect(names(table)))


