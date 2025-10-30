from langchain_core.prompts import PromptTemplate
import os
import dashscope
from dashscope import Generation

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        # Initialize the prompt based on role and other info
        self.prompt_template = self.create_prompt_template()
        
        # 使用阿里云百炼 DashScope API
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.model_name = os.getenv('DASHSCOPE_MODEL_NAME', 'qwen-plus')
        
        # 设置 DashScope API Key
        dashscope.api_key = self.api_key

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            templates = f"""
                你是一个由多学科医疗专业人员组成的医疗团队。
                你将收到一位患者的医疗报告，该患者已由心脏科医生、心理医生和肺科医生诊断。
                任务：审查患者从心脏科医生、心理医生和肺科医生获得的医疗报告，分析并列出患者可能存在的3个健康问题。
                请仅返回患者可能存在的3个健康问题的要点列表，并为每个问题提供原因说明。
                
                心脏科医生报告：{self.extra_info.get('cardiologist_report', '')}
                心理医生报告：{self.extra_info.get('psychologist_report', '')}
                肺科医生报告：{self.extra_info.get('pulmonologist_report', '')}
            """
        else:
            templates = {
                "Cardiologist": """
                    你是一名心脏科医生。你将收到一位患者的医疗报告。
                    任务：审查患者的心脏检查结果，包括心电图（ECG）、血液检查、动态心电图监测结果和超声心动图。
                    重点：确定是否有任何细微的心脏问题迹象可以解释患者的症状。排除任何潜在的心脏疾病，如心律失常或结构异常，这些在常规检查中可能被遗漏。
                    建议：提供关于需要进一步的心脏检查或监测的指导，以确保没有隐藏的心脏相关问题。如果发现心脏问题，请提出潜在的管理策略。
                    请仅返回患者症状的可能原因和建议的下一步措施。
                    医疗报告：{medical_report}
                """,
                "Psychologist": """
                    你是一名心理医生。你将收到一位患者的医疗报告。
                    任务：审查患者的报告并提供心理评估。
                    重点：识别可能影响患者心理健康的任何潜在心理健康问题，如焦虑、抑郁或创伤。
                    建议：提供如何解决这些心理健康问题的指导，包括心理治疗、咨询或其他干预措施。
                    请仅返回可能的心理健康问题和建议的下一步措施。
                    患者报告：{medical_report}
                """,
                "Pulmonologist": """
                    你是一名肺科医生。你将收到一位患者的医疗报告。
                    任务：审查患者的报告并提供肺部评估。
                    重点：识别可能影响患者呼吸的任何潜在呼吸系统问题，如哮喘、慢性阻塞性肺病（COPD）或肺部感染。
                    建议：提供如何解决这些呼吸系统问题的指导，包括肺功能测试、影像学检查或其他干预措施。
                    请仅返回可能的呼吸系统问题和建议的下一步措施。
                    患者报告：{medical_report}
                """
            }
            templates = templates[self.role]
        return PromptTemplate.from_template(templates)
    
    def run(self):
        print(f"{self.role} is running...")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        try:
            # 使用阿里云 DashScope API 调用
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=0
            )
            
            # 检查响应状态
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"Error: {response.code} - {response.message}")
                return None
                
        except Exception as e:
            print("Error occurred:", e)
            return None

# Define specialized agent classes
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        extra_info = {
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "pulmonologist_report": pulmonologist_report
        }
        super().__init__(role="MultidisciplinaryTeam", extra_info=extra_info)
