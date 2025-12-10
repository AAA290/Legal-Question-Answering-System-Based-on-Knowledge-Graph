import json
import os

from django.shortcuts import render,HttpResponse,redirect
from django.http import JsonResponse
from qa import models
import httpx
from django.http import JsonResponse
from openai import OpenAI
import logging

logging.basicConfig(filename="qa_test.log",
                    filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.DEBUG)

# Create your views here.
def login(request):
    return render(request,"login.html")

def login_form(request):
    if request.method == 'POST':
        name = request.POST.get('account')
        password = request.POST.get('password')
        try:
            obj = models.UserInfo.objects.get(name = name,password=password)
            request.session['userid'] = obj.id  # 将用户对象保存到会话中
            return render(request,"index.html",{"user":obj})
        except models.UserInfo.DoesNotExist:
            # 如果用户不存在，返回错误页面
            return render(request, 'error_page.html', {'error': '用户不存在'})
        # obj = models.UserInfo.objects.get(name = name,password=password)
    else:
        obj_id = request.session.get('userid')
        obj = models.UserInfo.objects.get(id=obj_id)
        return render(request,"index.html",{"user":obj})

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        password = request.POST.get('password')
        obj = models.UserInfo.objects.create(name=name,password=password)
    return render(request,"index.html",{"user":obj})

async def async_qa(request):
    if request.method == "POST":
        # 解析请求体中的 JSON 数据
        try:
            body = json.loads(request.body)
            question = body.get("question")
            if not question:
                return JsonResponse({"error": "question 字段不能为空"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "无效的 JSON 数据"}, status=400)

        # 转发到 FastAPI
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(
                    "http://127.0.0.1:8000/api/qa",
                    json={"question": question},
                    timeout=30.0
                )
                response.raise_for_status()
                # print(response.json())
                return JsonResponse(response.json())
            except httpx.HTTPError as e:
                return JsonResponse(
                    {"error": f"API请求失败: {str(e)}"},
                    status=503
                )
            finally:
                await client.aclose()  # 必须手动关闭
    else:
        return render(request, 'qa.html')

def home(request):
    obj_id = request.session.get('userid')
    if not obj_id:
        return render(request, 'error_page.html', {'error': '用户未登录'})
    obj = models.UserInfo.objects.filter(id=obj_id).first()
    if not obj:
        return render(request, 'error_page.html', {'error': '用户不存在'})
    return render(request,"index.html",{"user":obj})

async def crime_prediction(request):
    """处理前端请求，返回罪名预测结果"""
    if request.method == "POST":
        # 解析请求体中的 JSON 数据
        try:
            body = json.loads(request.body)
            fact = body.get("fact")
            if not fact:
                return JsonResponse({"error": "fact 字段不能为空"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "无效的 JSON 数据"}, status=400)

        # 转发到 FastAPI
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(
                    "http://127.0.0.1:8000/api/crimes_prediction",
                    json={"fact": fact},
                    timeout=30.0
                )
                response.raise_for_status()
                return JsonResponse(response.json())
            except httpx.HTTPError as e:
                return JsonResponse(
                    {"error": f"API请求失败: {str(e)}"},
                    status=503
                )
    else:
        return render(request, 'prediction.html')


async def flowchart(request):
    if request.method == "POST":
        # 解析请求体中的 JSON 数据
        try:
            body = json.loads(request.body)
            question = body.get("question")
            if not question:
                return JsonResponse({"error": "question 字段不能为空"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "无效的 JSON 数据"}, status=400)

        # 转发到 FastAPI
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(
                    "http://127.0.0.1:8000/api/flowchart",
                    json={"question": question},
                    timeout=30.0
                )
                response.raise_for_status()
                return JsonResponse(response.json())
            except httpx.HTTPError as e:
                return JsonResponse(
                    {"error": f"API请求失败: {str(e)}"},
                    status=503
                )
    else:
        return render(request,'flowchart.html')

def comparison(request):
    return render(request,'comparison.html')

def qa_gpt(request):
    if request.method == "POST":
        # 解析请求体中的 JSON 数据
        try:
            body = json.loads(request.body)
            question = body.get("question")
            if not question:
                return JsonResponse({"error": "question 字段不能为空"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "无效的 JSON 数据"}, status=400)

        try:
            # os.environ["http_proxy"] = "http://localhost:7890"
            # os.environ["https_proxy"] = "http://localhost:7890"
            proxies = "http://127.0.0.1:7890"
            # 这里设置代理以防止APIKEY ERROR
            client = OpenAI(
                api_key = os.environ.get("chatanywhere_API_KEY"),  # 从环境变量读取api key
                base_url="https://api.chatanywhere.tech/v1",
                http_client=httpx.Client(proxies=proxies)
                #设置了trust_env=False后配置这个就没有用了，os.environ也没有用，最后还是靠开vpn才没有报错api connectionerror
            )
            # print(os.environ.get("chatanywhere_API_KEY"))
            response = client.chat.completions.create(
                # model="deepseek-reasoner",  #要付费
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            # print(response.model_dump())
            return JsonResponse(response.model_dump())   #这个response.model_dump()处理成json格式也很重要
        except httpx.HTTPError as e:
            return JsonResponse(
                {"error": f"API请求失败: {str(e)}"},
                status=503
            )
    else:
        return render(request,'comparison.html')

async def kgqa(request):
    if request.method == "POST":
        # 解析请求体中的 JSON 数据
        try:
            body = json.loads(request.body)
            question = body.get("question")
            if not question:
                return JsonResponse({"error": "question 字段不能为空"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "无效的 JSON 数据"}, status=400)

        # 转发到 FastAPI
        async with httpx.AsyncClient(trust_env=False) as client:  # trust_env=False就不会被代理影响了，真的很重要
            try:
                response = await client.post(
                    "http://127.0.0.1:8000/api/qa",
                    json={"question": question},
                    timeout=30.0
                )
                response.raise_for_status()
                logging.info(response.json)
                return JsonResponse(response.json())
            except httpx.HTTPError as e:
                return JsonResponse(
                    {"error": f"API请求失败: {str(e)}"},
                    status=503
                )
    else:
        return render(request, 'comparison.html')