"""lqa_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from qa import views as qa_views
from kg_visual import views as kg_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('login/',qa_views.login),
    path('login_form/',qa_views.login_form),
    path('register/',qa_views.register),

    path('index/',qa_views.home),
    path('qa/',qa_views.async_qa),

    #知识图谱可视化
    # 图谱首页，加载 graph.html 模板
    path('graph/', kg_views.graph_view, name='graph'),
    # 根据罪名查询对应图谱数据（返回 JSON 格式，供前端 ajax 调用）
    path('query/', kg_views.query_crime, name='query_crime'),
    # 根据节点 ID 扩展图谱（返回 JSON 格式）
    path('expand/', kg_views.expand_node, name='expand_node'),

    #罪名预测
    path("crime_prediction/", qa_views.crime_prediction, name="crime_prediction"),
    #流程图
    path("flowchart/",qa_views.flowchart),
    #对比
    path("comparison/",qa_views.comparison),
    path("qa_gpt/",qa_views.qa_gpt),
    path("qa_kgqa/",qa_views.kgqa)
]
