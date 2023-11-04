import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, color, exposure, morphology, filters


def find_contours(image):
    # Zwiększenie kontrastu obrazu
    contrast_image = exposure.adjust_sigmoid(image, gain=4, cutoff=0.40)

    red_channel = image[:, :, 0]  # Czerwony kanał
    green_channel = image[:, :, 1]  # Zielony kanał
    blue_channel = image[:, :, 2]  # Niebieski kanał

    brightness = (red_channel.mean() + green_channel.mean() + blue_channel.mean()) / 3

    # Ustal wartości percentyli dla jasności obrazu
    if brightness > 120:
        intensityP = 1
        intensityK = 25
    else:
        intensityP = 1
        intensityK = 50
    pp, pk = np.percentile(contrast_image, (intensityP, intensityK))

    # Przeskaluj intensywność obrazu
    image_rescaled = exposure.rescale_intensity(contrast_image, in_range=(pp, pk))

    # Konwersja do przestrzeni barw HSV
    hsv_image = color.rgb2hsv(image_rescaled)

    # Utwórz obraz czarno-biały na podstawie kanału wartości (V)
    black_white = 1 - hsv_image[:, :, 2]

    # Progowanie obrazu, aby wyodrębnić obszary samolotów
    threshold_value = filters.threshold_otsu(black_white)
    thresholded_image = black_white > threshold_value

    for i in range(3):
        thresholded_image = morphology.dilation(thresholded_image, morphology.disk(3))

    # thresholded_image = morphology.erosion(thresholded_image, morphology.disk(1))

    # Wygładź obraz filtrem Gaussa
    smoothed_image = filters.gaussian(thresholded_image, sigma=1.5)

    # Zastosuj operacje morfologiczne, aby wyostrzyć i zmienić kształt konturów
    smoothed_image = morphology.closing(smoothed_image, morphology.disk(4))
    # Znajdź kontury na przetworzonym obrazie
    contours = measure.find_contours(smoothed_image, 0.5)

    # Przygotowanie kolorów konturów
    colors = plt.cm.jet(np.linspace(0, 1, len(contours)))  # Generowanie kolorów dla każdego konturu

    # Wyodrębnienie centroidów i oznaczenie ich białymi kółkami
    centroids = []
    for contour in contours:
        centroid = np.mean(contour, axis=0)
        centroids.append(centroid)

    # Konwersja do tablicy NumPy
    centroids = np.array(centroids)

    return contours, colors, centroids


def main():
    planes = ['planes/samolot00.jpg', 'planes/samolot01.jpg', 'planes/samolot02.jpg', 'planes/samolot03.jpg',
              'planes/samolot18.jpg', 'planes/samolot05.jpg', 'planes/samolot19.jpg', 'planes/samolot07.jpg',
              'planes/samolot08.jpg', 'planes/samolot09.jpg', 'planes/samolot10.jpg', 'planes/samolot11.jpg',
              'planes/samolot12.jpg', 'planes/samolot13.jpg', 'planes/samolot14.jpg', 'planes/samolot15.jpg',
              'planes/samolot16.jpg', 'planes/samolot17.jpg']

    planes_img = [io.imread(plane) for plane in planes]

    fig, axes = plt.subplots(6, 3, figsize=(30, 60))  # Tworzenie siatki 6x3 subplotów
    fig.tight_layout()

    for i, plane in enumerate(planes_img):
        contours, colors, centroids = find_contours(plane)

        axes[int(i / 3), i % 3].imshow(plane)  # Wyświetlenie obrazu na subplotcie
        axes[int(i / 3), i % 3].axis('off')  # Wyłączenie osi na subplotcie

        for j, contour in enumerate(contours):
            axes[int(i / 3), i % 3].plot(contour[:, 1], contour[:, 0], linewidth=4, c=colors[j])

        # Oznaczenie centroidów jako białe kółka
        axes[int(i / 3), i % 3].plot(centroids[:, 1], centroids[:, 0], 'wo', markersize=5)

    # plt.show()
    plt.savefig('plot.pdf')


if __name__ == '__main__':
    main()
