from  DeepImageSearch import Index,LoadData,SearchImage
# image_list=LoadData().from_folder(["data"])
image_list=LoadData().from_folder(['test'])
print(len(image_list))
Index(image_list).Start()
# SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)
# SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)
SearchImage().plot_similar_images('fire.25.png')


