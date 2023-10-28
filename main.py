import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, io, color, exposure, morphology, filters


def find_contours(image):
    # Ustal wartości percentyli dla jasności obrazu
    intensityP = 1
    intensityK = 25
    pp, pk = np.percentile(image, (intensityP, intensityK))

    # Przeskaluj intensywność obrazu
    image_rescaled = exposure.rescale_intensity(image, in_range=(pp, pk))

    # Wygładź obraz filtrem Gaussa
    smoothed_image = filters.gaussian(image_rescaled, sigma=1.5)

    # Konwersja do przestrzeni barw HSV
    hsv_image = color.rgb2hsv(smoothed_image)

    # Utwórz obraz czarno-biały na podstawie kanału wartości (V)
    black_white = 1 - hsv_image[:, :, 2]

    # Progowanie obrazu, aby wyodrębnić obszary samolotów
    threshold_value = filters.threshold_otsu(black_white)
    thresholded_image = black_white > threshold_value

    # Zastosuj operacje morfologiczne, aby wyostrzyć i zmienić kształt konturów
    processed_image = morphology.closing(thresholded_image, morphology.disk(3))

    # Znajdź kontury na przetworzonym obrazie
    contours = measure.find_contours(processed_image, 0.3)

    # Przygotowanie kolorów konturów
    colors = plt.cm.jet(np.linspace(0, 1, len(contours)))  # Generowanie kolorów dla każdego konturu

    return contours, colors  # Zwrócenie konturów i przypisanych kolorów


def main():
    planes = ['planes/samolot00.jpg', 'planes/samolot01.jpg', 'planes/samolot02.jpg', 'planes/samolot03.jpg',
              'planes/samolot04.jpg', 'planes/samolot05.jpg', 'planes/samolot06.jpg', 'planes/samolot07.jpg',
              'planes/samolot08.jpg', 'planes/samolot09.jpg', 'planes/samolot10.jpg', 'planes/samolot11.jpg',
              'planes/samolot12.jpg', 'planes/samolot13.jpg', 'planes/samolot14.jpg', 'planes/samolot15.jpg',
              'planes/samolot16.jpg', 'planes/samolot17.jpg']
    # planes = ['planes/samolot02.jpg']
    planes_img = [io.imread(plane) for plane in planes]

    fig, axes = plt.subplots(6, 3, figsize=(30, 60))  # Tworzenie siatki 6x3 subplotów
    fig.tight_layout()

    for i, plane in enumerate(planes_img):
        contours, colors = find_contours(plane)

        axes[int(i / 3), i % 3].imshow(plane)  # Wyświetlenie obrazu na subplotcie
        axes[int(i / 3), i % 3].axis('off')  # Wyłączenie osi na subplotcie

        for j, contour in enumerate(contours):
            axes[int(i / 3), i % 3].plot(contour[:, 1], contour[:, 0], linewidth=2, c=colors[j])

    # plt.show()
    plt.savefig('plot.pdf')


if __name__ == '__main__':
    main()
