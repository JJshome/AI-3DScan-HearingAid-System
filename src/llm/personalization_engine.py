"""
개인화 엔진 모듈

사용자의 청력 데이터, 사용 패턴 및 선호도를 분석하여 맞춤형 추천과 조언을 제공합니다.
LLM 기반의 분석을 통해 사용자 경험을 개선합니다.
"""
import logging
import time
import asyncio
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union, Set

logger = logging.getLogger(__name__)

class PersonalizationEngine:
    """
    개인화 엔진 클래스
    
    사용자의 청력 데이터와 보청기 사용 패턴을 분석하여 맞춤형 조언과
    최적의 설정을 제공하는 LLM 기반 개인화 시스템
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        개인화 엔진 초기화
        
        Args:
            config: 선택적 설정 매개변수
        """
        self.config = config or {}
        
        # 개인화 설정
        self.learning_rate = self.config.get('learning_rate', 0.05)
        self.min_data_points = self.config.get('min_data_points', 10)
        self.prediction_horizon_days = self.config.get('prediction_horizon_days', 30)
        self.max_history_days = self.config.get('max_history_days', 365)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        
        # 청력 분석 모델 설정 (실제로는 더 복잡함)
        self.hearing_profile_model = None
        self.usage_pattern_model = None
        self.feedback_model = None
        
        # 사용자 데이터 캐시
        self.user_data_cache = {}
        self.recommendation_history = {}
        
        logger.info(f"개인화 엔진 초기화됨: 학습률={self.learning_rate}, 예측기간={self.prediction_horizon_days}일")
    
    async def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        사용자 프로필 로드
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            Dict[str, Any]: 사용자 프로필 데이터
        """
        logger.info(f"사용자 프로필 로드 중: {user_id}")
        
        # 실제 구현에서는 데이터베이스에서 사용자 프로필 로드
        # 예시 코드:
        # profile = await db.users.find_one({"_id": user_id})
        
        # 테스트 데이터 시뮬레이션
        await asyncio.sleep(0.3)
        
        # 캐시 확인
        if user_id in self.user_data_cache:
            logger.info(f"사용자 프로필 캐시 히트: {user_id}")
            return self.user_data_cache[user_id]
        
        # 시뮬레이션된 사용자 프로필
        profile = {
            "user_id": user_id,
            "personal_info": {
                "age": 68,
                "gender": "male",
                "language": "ko",
                "occupation": "retired",
                "lifestyle": "active",
                "tech_savviness": "moderate"
            },
            "hearing_data": {
                "diagnosis_date": "2024-02-15",
                "hearing_loss_type": "sensorineural",
                "hearing_loss_level": "moderate",
                "left_ear": {
                    "low_freq_loss": 35,  # dB
                    "mid_freq_loss": 45,
                    "high_freq_loss": 60
                },
                "right_ear": {
                    "low_freq_loss": 40,
                    "mid_freq_loss": 50,
                    "high_freq_loss": 65
                },
                "speech_recognition": 72,  # %
                "tinnitus": True,
                "noise_sensitivity": "moderate"
            },
            "device_info": {
                "model": "HiRes-3DX-500",
                "purchase_date": "2024-03-10",
                "firmware_version": "2.4.1",
                "last_calibration": "2025-04-20",
                "features": [
                    "noise_reduction",
                    "directional_microphones",
                    "bluetooth_connectivity",
                    "rechargeable_battery",
                    "telecoil",
                    "smartphone_app"
                ]
            },
            "preferences": {
                "volume_level": 7,
                "preferred_environments": ["quiet", "conversation"],
                "challenging_environments": ["restaurant", "crowd"],
                "sound_preferences": {
                    "bass_emphasis": True,
                    "clarity_preference": "high",
                    "loudness_tolerance": "moderate"
                },
                "notification_preferences": {
                    "battery_alerts": True,
                    "maintenance_reminders": True,
                    "firmware_updates": True
                }
            },
            "created_at": "2024-03-10T09:25:31Z",
            "updated_at": "2025-05-05T14:30:22Z"
        }
        
        # 캐시에 저장
        self.user_data_cache[user_id] = profile
        
        logger.info(f"사용자 프로필 로드 완료: {user_id}")
        return profile
    
    async def load_usage_patterns(self, 
                                user_id: str, 
                                days: Optional[int] = None) -> Dict[str, Any]:
        """
        사용자의 보청기 사용 패턴 로드
        
        Args:
            user_id: 사용자 식별자
            days: 데이터를 로드할 이전 일수 (기본값: max_history_days)
            
        Returns:
            Dict[str, Any]: 사용 패턴 데이터
        """
        days = days or self.max_history_days
        logger.info(f"사용 패턴 로드 중: {user_id}, 기간={days}일")
        
        # 실제 구현에서는 데이터베이스에서 사용 기록 로드
        # 예시 코드:
        # start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        # usage_logs = await db.usage_logs.find({"user_id": user_id, "timestamp": {"$gte": start_date}})
        
        # 시뮬레이션된 사용 패턴
        await asyncio.sleep(0.5)
        
        # 가상의 사용 시간 생성
        current_date = datetime.datetime.now()
        usage_data = []
        
        for day in range(min(days, 90)):  # 최대 90일 데이터 시뮬레이션
            date = current_date - datetime.timedelta(days=day)
            date_str = date.strftime("%Y-%m-%d")
            
            # 주중/주말에 따라 다른 패턴 생성
            is_weekend = date.weekday() >= 5
            
            # 평일/주말 평균 사용 시간 (시간)
            avg_daily_usage = 8.5 if not is_weekend else 6.2
            
            # 약간의 랜덤성 추가
            import random
            daily_usage = max(0, avg_daily_usage + random.uniform(-1.5, 1.5))
            
            # 환경별 사용 시간
            environments = {
                "home": daily_usage * (0.6 if not is_weekend else 0.8),
                "outdoor": daily_usage * (0.2 if not is_weekend else 0.15),
                "public_space": daily_usage * (0.1 if not is_weekend else 0.05),
                "work": daily_usage * (0.1 if not is_weekend else 0)
            }
            
            # 프로그램별 사용 시간
            programs = {
                "general": daily_usage * 0.5,
                "noisy": daily_usage * 0.2,
                "music": daily_usage * 0.1,
                "conversation": daily_usage * 0.2
            }
            
            usage_data.append({
                "date": date_str,
                "total_hours": round(daily_usage, 1),
                "environments": {k: round(v, 1) for k, v in environments.items()},
                "programs": {k: round(v, 1) for k, v in programs.items()},
                "volume_adjustments": int(random.uniform(2, 8)),
                "program_switches": int(random.uniform(1, 5)),
                "connectivity_usage": {
                    "phone_calls": int(random.uniform(0, 3)),
                    "music_streaming": int(random.uniform(0, 2)) if random.random() > 0.3 else 0,
                    "tv_streaming": int(random.uniform(0, 2)) if random.random() > 0.4 else 0
                },
                "battery_cycles": 1 if random.random() > 0.2 else 0
            })
        
        # 통계 계산
        avg_daily_hours = sum(day["total_hours"] for day in usage_data) / len(usage_data)
        
        program_totals = {}
        for day in usage_data:
            for program, hours in day["programs"].items():
                program_totals[program] = program_totals.get(program, 0) + hours
        
        program_preferences = {
            program: round(hours / sum(program_totals.values()) * 100, 1)
            for program, hours in program_totals.items()
        }
        
        # 요약 통계
        usage_summary = {
            "user_id": user_id,
            "period_days": len(usage_data),
            "avg_daily_usage_hours": round(avg_daily_hours, 1),
            "weekday_avg_hours": round(sum(day["total_hours"] for day in usage_data if datetime.datetime.strptime(day["date"], "%Y-%m-%d").weekday() < 5) / 
                            sum(1 for day in usage_data if datetime.datetime.strptime(day["date"], "%Y-%m-%d").weekday() < 5), 1),
            "weekend_avg_hours": round(sum(day["total_hours"] for day in usage_data if datetime.datetime.strptime(day["date"], "%Y-%m-%d").weekday() >= 5) / 
                            max(1, sum(1 for day in usage_data if datetime.datetime.strptime(day["date"], "%Y-%m-%d").weekday() >= 5)), 1),
            "program_preferences": program_preferences,
            "total_volume_adjustments": sum(day["volume_adjustments"] for day in usage_data),
            "avg_volume_adjustments_per_day": round(sum(day["volume_adjustments"] for day in usage_data) / len(usage_data), 1),
            "consistency_score": round(random.uniform(65, 95), 1),  # % 일관성 점수
            "daily_data": usage_data
        }
        
        logger.info(f"사용 패턴 로드 완료: {user_id}, {len(usage_data)}일 데이터")
        return usage_summary
    
    async def analyze_hearing_trends(self, 
                                   user_id: str,
                                   include_predictions: bool = True) -> Dict[str, Any]:
        """
        사용자의 청력 변화 추세 분석
        
        Args:
            user_id: 사용자 식별자
            include_predictions: 미래 예측 포함 여부
            
        Returns:
            Dict[str, Any]: 청력 변화 분석 결과
        """
        logger.info(f"청력 변화 추세 분석 중: {user_id}")
        
        # 사용자 프로필 로드
        profile = await self.load_user_profile(user_id)
        
        # 청력 검사 기록 로드 (실제로는 데이터베이스에서 로드)
        # 예시 코드:
        # hearing_tests = await db.hearing_tests.find({"user_id": user_id}).sort("date", -1)
        
        # 시뮬레이션된 청력 검사 기록
        await asyncio.sleep(0.5)
        
        current_date = datetime.datetime.now()
        base_date = datetime.datetime.strptime(profile["hearing_data"]["diagnosis_date"], "%Y-%m-%d")
        
        # 사용자 청력 기본값
        base_left = profile["hearing_data"]["left_ear"]
        base_right = profile["hearing_data"]["right_ear"]
        
        # 최근 2년 동안의 청력 검사 기록 시뮬레이션
        tests = []
        intervals = [360, 270, 180, 90, 30]  # 간격일 (오래된 것부터)
        
        for days_ago in intervals:
            test_date = current_date - datetime.timedelta(days=days_ago)
            if test_date < base_date:
                continue
                
            # 약간의 변화 시뮬레이션
            import random
            
            # 기본적으로 약간의 청력 저하 (특히 고주파수)
            days_since_base = (test_date - base_date).days
            years_factor = days_since_base / 365
            
            # 저주파/중주파/고주파 청력 저하 계수 (연간 기준)
            deterioration_factors = {
                "low_freq": 1.0,
                "mid_freq": 1.5,
                "high_freq": 2.0
            }
            
            left_ear = {
                "low_freq_loss": base_left["low_freq_loss"] + years_factor * deterioration_factors["low_freq"] + random.uniform(-1, 1),
                "mid_freq_loss": base_left["mid_freq_loss"] + years_factor * deterioration_factors["mid_freq"] + random.uniform(-1, 1),
                "high_freq_loss": base_left["high_freq_loss"] + years_factor * deterioration_factors["high_freq"] + random.uniform(-1.5, 1.5)
            }
            
            right_ear = {
                "low_freq_loss": base_right["low_freq_loss"] + years_factor * deterioration_factors["low_freq"] + random.uniform(-1, 1),
                "mid_freq_loss": base_right["mid_freq_loss"] + years_factor * deterioration_factors["mid_freq"] + random.uniform(-1, 1),
                "high_freq_loss": base_right["high_freq_loss"] + years_factor * deterioration_factors["high_freq"] + random.uniform(-1.5, 1.5)
            }
            
            speech_recognition = max(0, min(100, profile["hearing_data"]["speech_recognition"] - years_factor * 1.5 + random.uniform(-2, 2)))
            
            tests.append({
                "date": test_date.strftime("%Y-%m-%d"),
                "left_ear": {k: round(v, 1) for k, v in left_ear.items()},
                "right_ear": {k: round(v, 1) for k, v in right_ear.items()},
                "speech_recognition": round(speech_recognition, 1),
                "audiologist_notes": f"정기 검진 {days_ago}일 전"
            })
        
        # 최신 검사 결과 순으로 정렬
        tests.sort(key=lambda x: x["date"], reverse=True)
        
        # 변화 계산
        if len(tests) >= 2:
            latest = tests[0]
            earliest = tests[-1]
            
            changes = {
                "left_ear": {
                    key: round(latest["left_ear"][key] - earliest["left_ear"][key], 1)
                    for key in latest["left_ear"]
                },
                "right_ear": {
                    key: round(latest["right_ear"][key] - earliest["right_ear"][key], 1)
                    for key in latest["right_ear"]
                },
                "speech_recognition": round(latest["speech_recognition"] - earliest["speech_recognition"], 1),
                "period_days": (datetime.datetime.strptime(latest["date"], "%Y-%m-%d") - 
                            datetime.datetime.strptime(earliest["date"], "%Y-%m-%d")).days
            }
        else:
            changes = {"insufficient_data": True}
        
        # 예측 생성
        predictions = None
        if include_predictions and len(tests) >= 2:
            latest = tests[0]
            
            # 선형 예측 (실제로는 더 복잡한 모델 사용)
            if "period_days" in changes and changes["period_days"] > 0:
                days_to_predict = self.prediction_horizon_days
                prediction_date = (datetime.datetime.strptime(latest["date"], "%Y-%m-%d") + 
                                datetime.timedelta(days=days_to_predict))
                
                # 일일 변화율 계산
                daily_changes = {
                    "left_ear": {
                        key: changes["left_ear"][key] / changes["period_days"]
                        for key in changes["left_ear"]
                    },
                    "right_ear": {
                        key: changes["right_ear"][key] / changes["period_days"]
                        for key in changes["right_ear"]
                    },
                    "speech_recognition": changes["speech_recognition"] / changes["period_days"]
                }
                
                # 예측값 계산
                predicted_left = {
                    key: latest["left_ear"][key] + daily_changes["left_ear"][key] * days_to_predict
                    for key in latest["left_ear"]
                }
                
                predicted_right = {
                    key: latest["right_ear"][key] + daily_changes["right_ear"][key] * days_to_predict
                    for key in latest["right_ear"]
                }
                
                predicted_speech = latest["speech_recognition"] + daily_changes["speech_recognition"] * days_to_predict
                
                predictions = {
                    "date": prediction_date.strftime("%Y-%m-%d"),
                    "left_ear": {k: round(v, 1) for k, v in predicted_left.items()},
                    "right_ear": {k: round(v, 1) for k, v in predicted_right.items()},
                    "speech_recognition": round(predicted_speech, 1),
                    "confidence": round(max(0.5, min(0.95, 0.85 - 0.05 * (days_to_predict / 180))), 2),
                    "prediction_basis": f"{len(tests)} 검사 기록 기반, {days_to_predict}일 예측"
                }
                
                # 예측 결과 분석
                predictions["analysis"] = {
                    "trend": "deteriorating" if (sum(predicted_left.values()) - sum(latest["left_ear"].values()) +
                                            sum(predicted_right.values()) - sum(latest["right_ear"].values())) / 6 > 2 else "stable",
                    "significant_changes": [
                        f"왼쪽 귀 고주파 손실 {predicted_left['high_freq_loss'] - latest['left_ear']['high_freq_loss']:.1f}dB 증가 예상"
                        if predicted_left['high_freq_loss'] - latest['left_ear']['high_freq_loss'] > 3 else None,
                        f"오른쪽 귀 고주파 손실 {predicted_right['high_freq_loss'] - latest['right_ear']['high_freq_loss']:.1f}dB 증가 예상"
                        if predicted_right['high_freq_loss'] - latest['right_ear']['high_freq_loss'] > 3 else None,
                        f"어음 인지도 {latest['speech_recognition'] - predicted_speech:.1f}% 감소 예상"
                        if latest['speech_recognition'] - predicted_speech > 3 else None
                    ],
                    "adjustment_needed": sum(predicted_left.values()) - sum(latest["left_ear"].values()) / 3 > 5 or
                                      sum(predicted_right.values()) - sum(latest["right_ear"].values()) / 3 > 5
                }
                
                # None 값 제거
                predictions["analysis"]["significant_changes"] = [c for c in predictions["analysis"]["significant_changes"] if c]
        
        result = {
            "user_id": user_id,
            "tests": tests,
            "total_tests": len(tests),
            "latest_test": tests[0] if tests else None,
            "changes": changes,
            "overall_trend": "deteriorating" if len(tests) >= 2 and 
                          sum(changes.get("left_ear", {}).values()) + 
                          sum(changes.get("right_ear", {}).values()) > 0 else "stable",
            "predictions": predictions
        }
        
        logger.info(f"청력 변화 추세 분석 완료: {user_id}, {len(tests)}개 검사 기록")
        return result
    
    async def generate_optimization_suggestions(self, 
                                            user_id: str,
                                            situation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        사용자 맞춤형 최적화 제안 생성
        
        Args:
            user_id: 사용자 식별자
            situation_type: 상황 유형 (일반, 소음 환경, 음악 감상 등)
            
        Returns:
            Dict[str, Any]: 최적화 제안 및 설정
        """
        logger.info(f"최적화 제안 생성 중: {user_id}, 상황={situation_type or '일반'}")
        
        # 사용자 데이터 로드
        profile = await self.load_user_profile(user_id)
        usage_patterns = await self.load_usage_patterns(user_id, days=60)
        
        # 실제 구현에서는 LLM을 사용하여 최적화 제안 생성
        # 예시 코드:
        # prompt = create_optimization_prompt(profile, usage_patterns, situation_type)
        # llm_response = await llm_client.complete(prompt)
        # suggestions = parse_llm_response(llm_response)
        
        # 시뮬레이션된 제안 (실제로는 LLM에서 생성됨)
        await asyncio.sleep(0.7)
        
        # 상황별 기본 최적화 제안
        base_suggestions = {
            "general": [
                "일상적인 환경에 맞게 소음 감소 수준을 '중간'으로 설정하세요.",
                "배터리 최적화를 위해 블루투스 기능을 필요할 때만 활성화하세요.",
                "주변 소리 인식 기능을 활성화하여 안전한 외출이 가능합니다."
            ],
            "noisy": [
                "소음이 많은 환경에서는 '고급 소음 감소' 기능을 활성화하세요.",
                "대화에 집중하기 위해 지향성 마이크를 '전방 집중' 모드로 설정하세요.",
                "저주파수 감쇠를 활성화하여 배경 소음을 줄이세요."
            ],
            "music": [
                "음악 감상 시 '음악 최적화' 프로필을 사용하세요.",
                "다이나믹 레인지를 최대로 설정하여 풍부한 음악 경험을 즐기세요.",
                "베이스 향상 기능을 활성화하여 더 풍부한 저음을 경험하세요."
            ],
            "conversation": [
                "대화 중에는 '음성 향상' 기능을 활성화하세요.",
                "조용한 환경에서는 마이크 감도를 높여 작은 소리도 잘 들리게 하세요.",
                "에코 감소 기능을 활성화하여 더 명확한 대화를 나눌 수 있습니다."
            ],
            "outdoor": [
                "바람 소리 감소 기능을 활성화하여 야외에서 더 잘 들을 수 있습니다.",
                "환경 적응형 설정을 '자동'으로 두어 변화하는 환경에 맞게 조절하세요.",
                "위치 기반 자동 설정 기능을 활성화하여 자주 가는 장소에서 최적의 설정을 유지하세요."
            ]
        }
        
        # 기본 제안 선택
        situation = situation_type or "general"
        if situation not in base_suggestions:
            situation = "general"
        
        suggestions = base_suggestions[situation].copy()
        
        # 사용자 맞춤형 제안 추가
        if profile["hearing_data"]["hearing_loss_level"] == "severe":
            suggestions.append("고도 난청에 맞게 압축 비율을 높여 작은 소리도 더 잘 들리게 하세요.")
        
        if profile["hearing_data"]["tinnitus"]:
            suggestions.append("이명 완화 기능을 활성화하여 배경 소리를 제공하세요.")
        
        if profile["personal_info"]["age"] > 70:
            suggestions.append("버튼과 컨트롤을 더 쉽게 조작할 수 있도록 앱 인터페이스를 '간편 모드'로 설정하세요.")
        
        if usage_patterns["avg_daily_usage_hours"] > 12:
            suggestions.append("장시간 사용 시 배터리 수명을 위해 에너지 절약 모드를 활성화하세요.")
        
        if "noisy" in usage_patterns["program_preferences"] and usage_patterns["program_preferences"]["noisy"] > 25:
            suggestions.append("소음 환경에서 많은 시간을 보내시는 것 같습니다. 정기적인 청력 검사를 통해 추가적인 소음 노출을 모니터링하세요.")
        
        # 맞춤형 설정 값 생성
        settings = {
            "noise_reduction": {
                "level": "high" if situation == "noisy" else "medium",
                "adaptive": True
            },
            "directional_mic": {
                "mode": "front" if situation == "conversation" or situation == "noisy" else "omni",
                "adaptive_steering": True
            },
            "equalizer": {
                "low_freq": situation == "music" and profile["preferences"]["sound_preferences"]["bass_emphasis"],
                "mid_freq": situation == "conversation",
                "high_freq": situation == "conversation" or profile["hearing_data"]["left_ear"]["high_freq_loss"] > 55
            },
            "features": {
                "wind_noise_reduction": situation == "outdoor",
                "feedback_suppression": True,
                "speech_enhancement": situation == "conversation" or situation == "noisy",
                "echo_cancellation": situation == "conversation",
                "tinnitus_masker": profile["hearing_data"]["tinnitus"]
            },
            "connectivity": {
                "auto_adapt_to_phone_calls": True,
                "stream_notification_sounds": profile["preferences"]["notification_preferences"]["battery_alerts"]
            }
        }
        
        # LLM 기반 개인화 제안 시뮬레이션
        personalized_advice = ""
        if situation == "general":
            personalized_advice = (
                f"{profile['personal_info']['age']}세 {profile['hearing_data']['hearing_loss_level']} "
                f"난청 사용자의 경우, 하루 평균 {usage_patterns['avg_daily_usage_hours']}시간 사용 패턴에 맞춰 "
                f"여러 환경에서 최적의 청취 경험을 위한 맞춤형 설정이 중요합니다. "
                f"특히 고주파수 손실({profile['hearing_data']['left_ear']['high_freq_loss']}dB)을 "
                f"보완하기 위한 주파수 조정과 상황별 프로그램 전환이 도움이 됩니다."
            )
        elif situation == "noisy":
            personalized_advice = (
                f"소음이 많은 환경에서는 지향성 마이크와 고급 소음 감소 기능의 조합이 효과적입니다. "
                f"특히 식당과 같은 환경에서는 전방 집중 모드로 설정하고, 필요에 따라 음성 명령 "
                f"\"집중 모드 켜줘\"를 통해 쉽게 전환할 수 있습니다."
            )
        elif situation == "music":
            personalized_advice = (
                f"음악 감상 시 최적의 경험을 위해 다이나믹 레인지를 확장하고, "
                f"사용자의 베이스 선호도에 맞춰 저주파수 향상을 활성화했습니다. "
                f"클래식 음악과 대중 음악에 대한 별도의 프리셋을 구성하여 "
                f"음악 장르에 따라 전환할 수 있습니다."
            )
        
        # 이전 제안과 비교하여 변경점 추적
        previous_suggestions = self.recommendation_history.get(user_id, {}).get(situation, [])
        changes_from_previous = []
        
        if previous_suggestions:
            # 이전 설정과 현재 설정 비교
            for suggestion in suggestions:
                if suggestion not in previous_suggestions:
                    changes_from_previous.append(f"새로운 제안: {suggestion}")
            
            for suggestion in previous_suggestions:
                if suggestion not in suggestions:
                    changes_from_previous.append(f"제거된 제안: {suggestion}")
        
        # 추천 기록 업데이트
        self.recommendation_history[user_id] = self.recommendation_history.get(user_id, {})
        self.recommendation_history[user_id][situation] = suggestions
        
        result = {
            "user_id": user_id,
            "situation": situation,
            "suggestions": suggestions,
            "settings": settings,
            "personalized_advice": personalized_advice,
            "changes_from_previous": changes_from_previous,
            "generated_at": datetime.datetime.now().isoformat(),
            "confidence_score": 0.92
        }
        
        logger.info(f"최적화 제안 생성 완료: {user_id}, {len(suggestions)}개 제안")
        return result
    
    async def predict_maintenance_needs(self, 
                                     user_id: str) -> Dict[str, Any]:
        """
        보청기 유지보수 필요성 예측
        
        Args:
            user_id: 사용자 식별자
            
        Returns:
            Dict[str, Any]: 유지보수 예측 및 권장사항
        """
        logger.info(f"유지보수 필요성 예측 중: {user_id}")
        
        # 사용자 데이터 로드
        profile = await self.load_user_profile(user_id)
        usage_patterns = await self.load_usage_patterns(user_id, days=90)
        
        # 실제 구현에서는 사용 패턴, 환경 데이터, 장치 연령 등을 분석
        # 예시 코드:
        # maintenance_model = load_maintenance_prediction_model()
        # prediction = maintenance_model.predict(usage_patterns, profile)
        
        # 시뮬레이션된 예측
        await asyncio.sleep(0.4)
        
        # 현재 날짜 및 기기 구매일 계산
        current_date = datetime.datetime.now()
        purchase_date = datetime.datetime.strptime(profile["device_info"]["purchase_date"], "%Y-%m-%d")
        days_since_purchase = (current_date - purchase_date).days
        
        # 마지막 보정일 계산
        last_calibration = datetime.datetime.strptime(profile["device_info"]["last_calibration"], "%Y-%m-%d")
        days_since_calibration = (current_date - last_calibration).days
        
        # 유지보수 필요성 평가 요소
        maintenance_factors = {
            "age_score": min(100, days_since_purchase / 7),  # 1주일 당 1점 (최대 100)
            "usage_score": min(100, usage_patterns["avg_daily_usage_hours"] * 7),  # 시간당 7점 (최대 100)
            "calibration_score": min(100, days_since_calibration / 3),  # 3일 당 1점 (최대 100)
            "environment_score": 50 + (20 if "challenging_environments" in profile["preferences"] and 
                              len(profile["preferences"]["challenging_environments"]) > 1 else 0),
            "volume_adjustment_score": min(100, usage_patterns["avg_volume_adjustments_per_day"] * 10)  # 조정당 10점
        }
        
        # 종합 점수 계산 (가중치 적용)
        weights = {
            "age_score": 0.2,
            "usage_score": 0.3,
            "calibration_score": 0.25,
            "environment_score": 0.15,
            "volume_adjustment_score": 0.1
        }
        
        maintenance_score = sum(score * weights[factor] for factor, score in maintenance_factors.items())
        
        # 권장사항 생성
        recommendations = []
        maintenance_schedule = []
        
        if days_since_calibration > 90:
            recommendations.append({
                "type": "critical",
                "action": "재보정",
                "reason": f"마지막 보정 이후 {days_since_calibration}일이 경과했습니다. 3개월마다 재보정을 권장합니다."
            })
            
            # 다음 2주 이내 권장
            next_calibration = current_date + datetime.timedelta(days=14)
            maintenance_schedule.append({
                "type": "calibration",
                "recommended_date": next_calibration.strftime("%Y-%m-%d"),
                "priority": "high"
            })
        
        if days_since_purchase > 180 and maintenance_score > 70:
            recommendations.append({
                "type": "important",
                "action": "청력 전문가 상담",
                "reason": "6개월 이상 사용했으며 사용 패턴 분석 결과 전문가 확인이 필요합니다."
            })
            
            # 다음 달 권장
            next_checkup = current_date + datetime.timedelta(days=30)
            maintenance_schedule.append({
                "type": "professional_checkup",
                "recommended_date": next_checkup.strftime("%Y-%m-%d"),
                "priority": "medium"
            })
        
        if "challenging_environments" in profile["preferences"] and len(profile["preferences"]["challenging_environments"]) > 0:
            recommendations.append({
                "type": "optimization",
                "action": "환경별 프로그램 설정 최적화",
                "reason": f"자주 {', '.join(profile['preferences']['challenging_environments'])} 환경에 있어 맞춤 설정이 필요합니다."
            })
        
        if usage_patterns["avg_volume_adjustments_per_day"] > 5:
            recommendations.append({
                "type": "usability",
                "action": "볼륨 프리셋 구성",
                "reason": "하루 평균 볼륨 조정 횟수가 많아 자주 사용하는 볼륨 수준을 프리셋으로 저장하면 편리합니다."
            })
        
        # 유지보수 예측 스케줄
        next_routine_checkup = current_date + datetime.timedelta(days=max(0, 180 - days_since_purchase) + 30)
        maintenance_schedule.append({
            "type": "routine_checkup",
            "recommended_date": next_routine_checkup.strftime("%Y-%m-%d"),
            "priority": "low"
        })
        
        # 펌웨어 업데이트 필요한지 확인
        firmware_update_needed = False
        if "firmware_version" in profile["device_info"]:
            # 실제로는 최신 펌웨어 버전과 비교
            current_firmware = profile["device_info"]["firmware_version"]
            latest_firmware = "2.4.2"  # 시뮬레이션된 최신 버전
            
            if current_firmware != latest_firmware:
                firmware_update_needed = True
                recommendations.append({
                    "type": "update",
                    "action": "펌웨어 업데이트",
                    "reason": f"현재 {current_firmware} 버전을 사용 중이나, 최신 {latest_firmware} 버전으로 업데이트하면 새로운 기능과 개선사항을 이용할 수 있습니다."
                })
        
        # 개인화된 LLM 유지보수 조언 시뮬레이션
        maintenance_advice = (
            f"{profile['personal_info']['age']}세 사용자의 {days_since_purchase}일 사용 기록을 분석한 결과, "
            f"현재 유지보수 점수는 {maintenance_score:.1f}/100점으로 "
            f"{'즉각적인 관리가 필요합니다' if maintenance_score > 75 else '정기적인 관리가 권장됩니다'}. "
            f"특히 {'청력 변화에 따른 재조정이 필요' if days_since_calibration > 90 else '최신 개인화 설정 적용이 도움됩니다'}."
        )
        
        result = {
            "user_id": user_id,
            "maintenance_score": round(maintenance_score, 1),
            "risk_level": "high" if maintenance_score > 75 else "medium" if maintenance_score > 50 else "low",
            "factors": {k: round(v, 1) for k, v in maintenance_factors.items()},
            "recommendations": recommendations,
            "maintenance_schedule": maintenance_schedule,
            "firmware_update_available": firmware_update_needed,
            "personalized_advice": maintenance_advice,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"유지보수 필요성 예측 완료: {user_id}, 점수={maintenance_score:.1f}")
        return result
    
    async def track_preference_changes(self, 
                                    user_id: str, 
                                    period_days: int = 90) -> Dict[str, Any]:
        """
        사용자 선호도 변화 추적
        
        Args:
            user_id: 사용자 식별자
            period_days: 분석 기간 (일)
            
        Returns:
            Dict[str, Any]: 선호도 변화 분석 결과
        """
        logger.info(f"선호도 변화 추적 중: {user_id}, 기간={period_days}일")
        
        # 사용 패턴 로드
        usage_patterns = await self.load_usage_patterns(user_id, days=period_days)
        
        # 실제 구현에서는 기간별 사용 패턴 비교 분석
        # 예시 코드:
        # monthly_patterns = split_usage_by_month(usage_patterns)
        # preference_changes = analyze_preference_trends(monthly_patterns)
        
        # 시뮬레이션된 선호도 변화 분석
        await asyncio.sleep(0.5)
        
        # 데이터를 월별로 분할
        daily_data = usage_patterns["daily_data"]
        daily_data.sort(key=lambda x: x["date"])
        
        # 최대 3개월 데이터 분석
        months = min(3, period_days // 30)
        month_size = len(daily_data) // months
        
        monthly_data = []
        for i in range(months):
            start_idx = i * month_size
            end_idx = (i + 1) * month_size if i < months - 1 else len(daily_data)
            
            month_chunk = daily_data[start_idx:end_idx]
            
            # 월별 통계 계산
            if not month_chunk:
                continue
                
            month_avg_usage = sum(day["total_hours"] for day in month_chunk) / len(month_chunk)
            
            # 프로그램별 사용 시간
            program_usage = {}
            for day in month_chunk:
                for program, hours in day["programs"].items():
                    program_usage[program] = program_usage.get(program, 0) + hours
            
            # 백분율로 변환
            program_preferences = {}
            total_hours = sum(program_usage.values())
            if total_hours > 0:
                program_preferences = {
                    program: round(hours / total_hours * 100, 1)
                    for program, hours in program_usage.items()
                }
            
            # 환경별 사용 시간
            environment_usage = {}
            for day in month_chunk:
                for env, hours in day["environments"].items():
                    environment_usage[env] = environment_usage.get(env, 0) + hours
            
            # 백분율로 변환
            environment_preferences = {}
            if total_hours > 0:
                environment_preferences = {
                    env: round(hours / total_hours * 100, 1)
                    for env, hours in environment_usage.items()
                }
            
            # 기타 패턴
            avg_volume_adjustments = sum(day["volume_adjustments"] for day in month_chunk) / len(month_chunk)
            
            start_date = month_chunk[0]["date"]
            end_date = month_chunk[-1]["date"]
            
            monthly_data.append({
                "period": f"{start_date} ~ {end_date}",
                "avg_daily_usage": round(month_avg_usage, 1),
                "program_preferences": program_preferences,
                "environment_preferences": environment_preferences,
                "avg_volume_adjustments": round(avg_volume_adjustments, 1),
                "days": len(month_chunk)
            })
        
        # 월별 변화 분석
        changes = []
        
        if len(monthly_data) >= 2:
            # 가장 최근 2개월 비교
            recent = monthly_data[-1]
            previous = monthly_data[-2]
            
            # 사용 시간 변화
            usage_change = recent["avg_daily_usage"] - previous["avg_daily_usage"]
            if abs(usage_change) > 0.5:
                changes.append({
                    "type": "usage_time",
                    "change": f"하루 평균 사용 시간이 {abs(usage_change):.1f}시간 {'증가' if usage_change > 0 else '감소'}했습니다.",
                    "significance": "high" if abs(usage_change) > 1.0 else "medium"
                })
            
            # 프로그램 선호도 변화
            for program in set(recent["program_preferences"].keys()).union(previous["program_preferences"].keys()):
                recent_pref = recent["program_preferences"].get(program, 0)
                previous_pref = previous["program_preferences"].get(program, 0)
                
                pref_change = recent_pref - previous_pref
                
                if abs(pref_change) > 5:
                    changes.append({
                        "type": "program_preference",
                        "change": f"'{program}' 프로그램 사용이 {abs(pref_change):.1f}% {'증가' if pref_change > 0 else '감소'}했습니다.",
                        "significance": "high" if abs(pref_change) > 10 else "medium"
                    })
            
            # 환경 선호도 변화
            for env in set(recent["environment_preferences"].keys()).union(previous["environment_preferences"].keys()):
                recent_pref = recent["environment_preferences"].get(env, 0)
                previous_pref = previous["environment_preferences"].get(env, 0)
                
                pref_change = recent_pref - previous_pref
                
                if abs(pref_change) > 5:
                    changes.append({
                        "type": "environment_preference",
                        "change": f"'{env}' 환경에서의 사용이 {abs(pref_change):.1f}% {'증가' if pref_change > 0 else '감소'}했습니다.",
                        "significance": "high" if abs(pref_change) > 10 else "medium"
                    })
        
        # LLM 기반 분석 시뮬레이션
        preference_analysis = ""
        if len(monthly_data) >= 2:
            # 가장 선호하는 프로그램 및 환경 추출
            if monthly_data[-1]["program_preferences"]:
                top_program = max(monthly_data[-1]["program_preferences"].items(), key=lambda x: x[1])
                
                preference_analysis = (
                    f"최근 {monthly_data[-1]['days']}일 간의 사용 패턴을 분석한 결과, "
                    f"'{top_program[0]}' 프로그램을 전체의 {top_program[1]:.1f}% 시간 동안 사용하여 가장 선호하는 것으로 나타났습니다. "
                )
                
                if changes:
                    significant_changes = [c for c in changes if c["significance"] == "high"]
                    if significant_changes:
                        preference_analysis += (
                            f"특히 주목할만한 변화로는 {significant_changes[0]['change']} "
                            f"이러한 변화를 고려하여 보청기 설정을 조정하면 더 나은 청취 경험을 제공할 수 있습니다."
                        )
        
        result = {
            "user_id": user_id,
            "period_days": period_days,
            "monthly_data": monthly_data,
            "significant_changes": changes,
            "preference_analysis": preference_analysis,
            "generated_at": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"선호도 변화 추적 완료: {user_id}, {len(changes)}개 유의미한 변화 발견")
        return result
