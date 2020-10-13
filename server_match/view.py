from django.http import HttpResponse
from test_pipeline import test
from django.shortcuts import render
import base64
import torch.cuda
from PIL import Image
cat2id = {0:'Bar', 1:'Line', 2:'Pie'}
Lock = False
def get_group(request):
    global Lock
    print("The method is: %s" %request.method)
    if not Lock:
        if request.method == 'POST':
            #print(request.FILES)
            #print(request.POST)
            print("Clean Cuda Cache")
            torch.cuda.empty_cache()
            Lock = True
            try:
                if 'file' in request.FILES.keys():
                    with open('static/target.png', 'wb') as fout:
                        for chunk in request.FILES['file'].chunks():
                            fout.write(chunk)
                if len(request.POST['min']) > 0:
                    min_value = float(request.POST['min'])
                    max_value = float(request.POST['max'])
                else:
                    min_value = None
                    max_value = None
                plot_area, image_painted, data, chart_data = test('static/target.png', min_value_official=min_value, max_value_official=max_value)
                print_data = ''
                if chart_data[0]==0:
                    if len(request.POST['min']) > 0:
                        min_value = float(request.POST['min'])
                        max_value = float(request.POST['max'])
                    else:
                        min_value = chart_data[3]
                        max_value = chart_data[4]
                    for k in range(len(data)):
                        for j in range(len(data[k])):
                            data[k][j] = round((max_value - min_value) * data[k][j] + min_value, 2)
                            print_data += ('%8.2f' % (data[k][j]))
                            print_data += ' '
                        print_data += '\n'
                if chart_data[0] == 1:
                    if len(request.POST['min']) > 0:
                        min_value = float(request.POST['min'])
                        max_value = float(request.POST['max'])
                    else:
                        min_value = chart_data[3]
                        max_value = chart_data[4]
                    for k in range(len(data)):
                        for j in range(len(data[k])):
                            data[k][j] = round((max_value - min_value) * data[k][j] + min_value, 2)
                            print_data += ('%8.2f' % (data[k][j]))
                            print_data += ' '
                        print_data += '\n'
                if chart_data[0]==2:
                    for k in range(len(data)):
                        data[k] /= 360
                    data = [round(x, 2) for x in data]
                    for k in range(len(data)):
                        print_data += ('%8.2f' % (data[k]))
                        print_data += ' '
                    min_value = 0
                    max_value = 1
                image_painted.save('static/target_draw.png')
                with open("static/target.png", "rb") as f:
                    base64_data = base64.b64encode(f.read())
                    str_format_ori = 'data:image/png;base64,' + base64_data.decode()
                with open("static/target_draw.png", "rb") as f:
                    base64_data = base64.b64encode(f.read())
                    str_format_tar = 'data:image/png;base64,' + base64_data.decode()
                context = {'data': print_data, 'image': str_format_ori, 'image_painted': str_format_tar, 'plot_area': plot_area, 'min2max': '%2f:%2f' %(min_value, max_value)}
                title2string = chart_data[2]
                if 1 in title2string.keys():
                    context['ValueAxisTitle'] = title2string[1]
                else:
                    context['ValueAxisTitle'] = "None"
                if 2 in title2string.keys():
                    context['ChartTitle'] = title2string[2]
                else:
                    context['ChartTitle'] = "None"
                if 3 in title2string.keys():
                    context['CategoryAxisTitle'] = title2string[3]
                else:
                    context['CategoryAxisTitle'] = "None"
                context['Type'] = cat2id[chart_data[0]]
            except:
                print('We met some errors!')
                Lock = False
                raise
            Lock = False
            return render(request, 'results.html', context)
        else:
            return render(request, 'upload.html')
    else:
        return render(request, 'onuse.html')

type2idFormal = {
    "Legend" : 0,
    "ValueAxisTitle" : 1,
    "ChartTitle" : 2,
    "CategoryAxisTitle" : 3,
    "PlotArea" : 4,
    "InnerPlotArea" : 5}