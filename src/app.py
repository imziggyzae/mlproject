import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(
    __name__,
    template_folder=str(project_root / 'templates')
)
app = application


@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.exception("Unhandled application error")
    return render_template('home.html', error="Something went wrong while processing your request. Please refresh and try again."), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        data = CustomData(
            age=float(request.form.get('age')),
            gender=request.form.get('gender'),
            daily_social_media_hours=float(request.form.get('daily_social_media_hours')),
            platform_usage=request.form.get('platform_usage'),
            sleep_hours=float(request.form.get('sleep_hours')),
            screen_time_before_sleep=float(request.form.get('screen_time_before_sleep')),
            academic_performance=float(request.form.get('academic_performance')),
            physical_activity=float(request.form.get('physical_activity')),
            social_interaction_level=request.form.get('social_interaction_level'),
            stress_level=float(request.form.get('stress_level')),
            anxiety_level=float(request.form.get('anxiety_level')),
            depression_label=float(request.form.get('depression_label')),
        )

        pred_df = data.get_data_as_data_frame()
        predictions = PredictPipeline().predict(pred_df)
        predicted_value = float(predictions[0])

        return render_template(
            'home.html',
            results=f"Predicted addiction level: {predicted_value}",
            predicted_value=predicted_value,
        )
    except Exception as e:
        app.logger.exception("Prediction request failed")
        return render_template('home.html', error=f"Prediction error: {e}")


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0')
