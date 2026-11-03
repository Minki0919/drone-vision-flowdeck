# 파일 역할: Python 패키지와 Gazebo 자원 설치 정보
# 파트: 필요한 라이브러리와 프로젝트 모듈 연결
from glob import glob
from setuptools import find_packages, setup


# 파트: 공통 경로·좌표계·설정값
package_name = "dynamic_drone"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.config")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
        ("share/" + package_name + "/models/crazyflie_2_1", glob("models/crazyflie_2_1/*")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="Dynamic Drone Team",
    maintainer_email="dynamic.drone@example.com",
    description="Mission control experiments for the Dynamic Drone challenge.",
    license="MIT",
)
