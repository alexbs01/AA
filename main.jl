import Pkg
Pkg.add("Images")
Pkg.add("FileIO")

using Images
using FileIO
using Statistics

# Define the path to the dataset folder
dataset_folder = "dataset/train/"

# Get a list of all subfolders in the dataset folder
subfolders = readdir(dataset_folder)



# Loop through each subfolder
for subfolder in subfolders
    # Check if the subfolder is "Very_Low" or "Very_High"
    if subfolder == "Very_Low" || subfolder == "Very_High"
        # Get the path to the current subfolder
        subfolder_path = joinpath(dataset_folder, subfolder)

        # Get a list of all image files in the subfolder
        image_files = filter(file ->
                isfile(joinpath(subfolder_path, file)), readdir(subfolder_path))

        # Loop through each image file
        for image_file in image_files
            # Get the path to the current image file
            image_path = joinpath(subfolder_path, image_file)

            # Load the image into memory
            image = load(image_path)

            # Append the image to the corresponding array
            if subfolder == "Very_Low"
                push!(very_low_images, image)
            elseif subfolder == "Very_High"
                push!(very_high_images, image)
            end
        end
    end
end

# Now the images are divided into the `very_low_images` and `very_high_images` arrays


# Calcular la media y la desviación estándar de
very_low_red_mean = [mean(red(image)) for image in very_low_images]
very_low_green_mean = [mean(green(image)) for image in very_low_images]
very_low_blue_mean = [mean(blue(image)) for image in very_low_images]

very_low_red_std = [std(red(image)) for image in very_low_images]
very_low_green_std = [std(green(image)) for image in very_low_images]
very_low_blue_std = [std(blue(image)) for image in very_low_images]

# Calcular la media y la desviación estándar de los canales de color para very_high_images
very_high_red_mean = [mean(red(image)) for image in very_high_images]
very_high_green_mean = [mean(green(image)) for image in very_high_images]
very_high_blue_mean = [mean(blue(image)) for image in very_high_images]

very_high_red_std = [std(red(image)) for image in very_high_images]
very_high_green_std = [std(green(image)) for image in very_high_images]
very_high_blue_std = [std(blue(image)) for image in very_high_images]
