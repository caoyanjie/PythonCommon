# _*_ coding: utf-8 _*_
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import InvalidElementStateException
from selenium.common.exceptions import NoSuchFrameException
import os
import time
import pickle
from log import Log


class WebdriverManager:
    def __init__(self, driver_name='', is_default=False, login_url=None, username=None, passwd=None):
        self.__cookies_file_path = '%s/Selenium/phantomjs_cookies.pkl' % os.environ["APPDATA"]
        self.__log_path = 'log.txt'
        self.__window_handles = []
        self.__log = Log()
        if driver_name:
            self.get_webdriver(driver_name, is_default, login_url=None, username=None, passwd=None)

    def get_webdriver(self, driver_name, is_default=False, login_url=None, username=None, passwd=None):
        if driver_name == 'phantomjs':                 # PhantomJS
            if is_default:
                dcap = dict(DesiredCapabilities.PHANTOMJS)
                dcap['phantomjs.page.settings.userAgent'] = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
                # dcap['phantomjs.page.settings.userAgent'] = ('Mozilla/5.0 (Matrix OS 2.0) AppleWebKit/537.36 (KHTML, like Gecko) Matrix Browser/2.0')
                self.__driver = webdriver.PhantomJS(desired_capabilities=dcap, service_log_path=os.devnull)
                self.__get_exists_cookies(login_url, username, passwd)
            else:
                self.__driver = webdriver.PhantomJS(service_log_path=os.devnull)
            self.__driver.set_page_load_timeout(5)
            self.__driver.maximize_window()
        elif driver_name == 'firefox':                 # Firefox browser
            if is_default:
                profile = webdriver.FirefoxProfile(self.__get_default_browser_profile('firefox'))
                self.__driver = webdriver.Firefox(firefox_profile=profile, log_path=os.devnull)
            else:
                self.__driver = webdriver.Firefox(log_path=os.devnull)
            self.__driver.set_page_load_timeout(5)
            self.__driver.maximize_window()
        elif driver_name == 'chrome':                  # Chrome browser
            if is_default:
                option = webdriver.ChromeOptions()
                option.add_argument('--user-data-dir=%s' % self.__get_default_browser_profile("chrome"))
                self.__driver = webdriver.Chrome(chrome_options=option, service_log_path=os.devnull)
            else:
                self.__driver = webdriver.Chrome(service_log_path=os.devnull)
            self.__driver.maximize_window()
        else:
            assert False, 'Unknown exists driver name'

        return self.__driver

    def get(self, url):
        """ 安全的打开一个网址，如果触发超时异常，忽略 """
        try:
            self.__driver.get(url)
        except:
            pass            

    def open_url_in_new_window(self, url):
        """ 新开一个标签页，打开网址 """
        try:
            self.__driver.execute_script('window.open("%s")' % url)
        except:
            pass

    def get_current_url(self):
        return self.__driver.current_url

    def get_current_window_handle(self):
        return self.__driver.current_window_handle

    def get_window_handles(self):
        return self.__driver.window_handles

    def open_link_in_new_window(self, link, css_class_name=None, warning_msg=''):
        if css_class_name is None:
            js = '''
                links = document.getElementsByTagName("a");
                for (var i in links) {
                    links[i].target = "_blank";
                }'''
        else:
            js = '''
                links = document.getElementsByTagName("a");
                for (var i in links) {
                    if (links[i].className == "%s") {
                        links[i].target = "_blank";
                        break;
                    }
                }''' % css_class_name
        self.__driver.execute_script(js)
        try:
            link.click()
        except:
            if warning_msg:
                self.write_log(warning_msg)

    def switch_to_window(self, window_handle):
        self.__driver.switch_to_window(window_handle)

    def switch_to_new_window(self):
        if self.__driver.current_window_handle not in self.__window_handles:
            self.__window_handles.append(self.__driver.current_window_handle)
            self.__log.show_log(u'当前窗口[%s]' % self.__driver.title)
        for i in self.__driver.window_handles:
            if i not in self.__window_handles:
                self.__driver.switch_to.window(i)
                self.__window_handles.append(i)
                self.__log.show_log(u'切换至新窗口[%s]' % self.__driver.title)
                return True
        self.__log.show_log(u'没新窗口可切换')
        return False

    #def switch_to_new_window_and_close_current(url, )
    def switch_to_new_window_use_current_url(self, need_close_current_window=True):
        old_windows = self.__driver.window_handles
        self.open_url_in_new_window(self.__driver.current_url)                   
        for i in self.__driver.window_handles:
            if i not in old_windows:
                if need_close_current_window:
                    self.__driver.close()
                self.__driver.switch_to_window(i)
                break
    
    def switch_to_previous_window_after_current_window_close_by_other(self):
        self.__window_handles.pop()
        self.__driver.switch_to.window(self.__window_handles[-1])

    def close_window_and_switch_back(self):
        if self.__driver.current_window_handle == self.__window_handles[-1]:
            self.__driver.close()
            self.__window_handles.pop()
            self.__driver.switch_to.window(self.__window_handles[-1])

    def switch_to_frame(self, frame_name):
        error = 0
        while True:
            try:
                self.__driver.switch_to_frame(frame_name)
                self.__log.show_log(u'已切换至iframe[%s]' % frame_name)
                break
            except NoSuchFrameException:
                error += 1
                if error > 10:
                    return -1
                time.sleep(0.2)

    # 安全的查找元素
    def get_element(self, css_selector, timeout=30):
        failed_times = 0
        while True:
            try:
                return self.__driver.find_element(By.CSS_SELECTOR, css_selector)
            except:
                failed_times += 1
                if failed_times > timeout:
                    #self.__driver.save_screenshot('%s/Desktop/error.png' % os.environ["HOMEPATH"])
                    #self.__write_log(u'css_selector<%s>找不到' % css_selector)
                    #self.__write_log(u'错误页面截图已保存在%s/Desktop/error.png' % os.environ["HOMEPATH"])
                    #self.__driver.refresh()
                    #self.__write_log('已刷新页面')
                    #failed_times = 0
                    self.__log.show_log(u'css_selector<%s>找不到' % css_selector)
                    return None
                time.sleep(1)
    
    def get_element_by_element(self, element, css_selector, timeout=30):
        if element is None:
            return None
        
        failed_times = 0
        while True:
            try:
                return element.find_element(By.CSS_SELECTOR, css_selector)
            except:
                failed_times += 1
                if failed_times > timeout:
                    self.__log.show_log(u'css_selector<%s>找不到' % css_selector)
                    return None
                time.sleep(1)

    # 安全的查找元素列表
    def get_elements(self, css_selector, timeout=30):
        failed_times = 0
        while True:
            elements = self.__driver.find_elements(By.CSS_SELECTOR, css_selector)
            if len(elements) > 0:
                return elements
            else:
                failed_times += 1
                if failed_times > timeout:
                    self.__log.show_log(u'css_selector<%s>找不到' % css_selector)
                    #self.__driver.save_screenshot('%s/Desktop/error.png' % os.environ["HOMEPATH"])
                    #self.__write_log(u'错误页面截图已保存在%s/Desktop/error.png' % os.environ["HOMEPATH"])
                    return None
                time.sleep(1)

    def get_elements_by_element(self, element, css_selector, timeout=30):
        if element is None:
            return None
        
        failed_times = 0
        while True:
            elements = element.find_elements(By.CSS_SELECTOR, css_selector)
            if len(elements):
                return elements
            else:
                failed_times += 1
                if failed_times > timeout:
                    self.__log.show_log(u'css_selector<%s>找不到' % css_selector)
                    return None
                time.sleep(1)

    def get_input_value(self, css_selector, get_element_timeout=30, get_value_timeout=5):
        input_element = self.get_element(css_selector, get_element_timeout)
        if input_element is None:
            return None
        error = 0
        step = 0.2
        while True:
            if input_element.tag_name == 'input':
                result = input_element.get_attribute('value')
            elif input_element.tag_name == 'select':
                result = input_element.get_attribute('title')
            if result:
                return result
            else:
                error += 1
                if error > (get_value_timeout/step):
                    return None
                time.sleep(step)

    def click(self, css_selector_or_element, after_sleep_time=0, warning_msg='', find_element_timeout=30):
        if type(css_selector_or_element) is str:
            element = self.get_element(css_selector_or_element, timeout=find_element_timeout)
        else:
            element = css_selector_or_element
        if element is None:
            self.__log.show_log('Element is None!')
            return False

        while True:
            try:
                element.click()
                if after_sleep_time != 0:
                    time.sleep(after_sleep_time)
                break
            except ElementNotVisibleException:
                self.__log.show_log('Exception: WebdriverManager.ElementNotVisibleException')
                time.sleep(0.2)
            except Exception:
                if warning_msg:
                    self.__log.show_log(warning_msg)
        return True

    def send_keys(self, css_selector_or_element, keys, find_element_timeout=30):
        if type(css_selector_or_element) is str:      # get element
            element = self.get_element(css_selector_or_element, timeout=find_element_timeout)
        else:                                   # element or None
            element = css_selector_or_element
        if element is None:
            self.__log.show_log('Element is None!')
            return False

        while True:
            try:
                element.clear()
                element.send_keys(keys)
                break
            except ElementNotVisibleException:
                time.sleep(0.2)
            except InvalidElementStateException:
                time.sleep(0.2)
                return False
        return True

    def select_option(self, css_selector, index_or_value):
        select = self.get_element(css_selector)
        if select is None:
            return None
        if type(index_or_value) is int:
            Select(select).select_by_index(index_or_value)
        elif type(index_or_value) is str:
            Select(select).select_by_value(index_or_value)

    def get_options(self, select_css_selector):
        select = self.get_element(select_css_selector)
        return Select(select).options

    def mouse_move_to_element_with_offset(self, to_element, x_offset, y_offset):
        ActionChains(self.__driver).move_to_element_with_offset(to_element, x_offset, y_offset).perform()

    def mouse_try_move_to_element_with_offset_and_click(self, to_element, x_offset_range, y_offset_range):
        x_step = to_element.size['width'] - 2
        y_step = to_element.size['height'] - 2
        old_url = self.__driver.current_url
        for x in range(x_offset_range):
            for y in range(y_offset_range):
                ActionChains(self.__driver).move_to_element_with_offset(to_element, x_step*x, y_step*y).perform()
                ActionChains(self.__driver).click()
                time.sleep(1)
                if self.__driver.current_url != old_url:
                    return True
        return False

    def mouse_click(self):
        ActionChains(self.__driver).click().perform()

    def mouse_context_click(self):
        pass

    def save_screenshot(self, screenshot_saved_path):
        self.__driver.save_screenshot(screenshot_saved_path)

    def accept_alert(self):
        self.__driver.switch_to_alert().accept()

    def close(self):
        self.__driver.close()

    def quit(self):
        self.__driver.quit()

    def __write_log(self, log):
        self.__log.show_log(log)
        with open(self.__log_path, 'a') as f:
            f.write(log + '\n')

    def __get_exists_cookies(self, login_url, username, passwd):
        if not os.path.isfile(self.__cookies_file_path):
            dir_path = self.__cookies_file_path[:self.__cookies_file_path.rfind('/')]
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            tmp_option = webdriver.ChromeOptions()
            tmp_option.add_argument('--user-data-dir=%s' % self.__get_default_browser_profile("chrome"))
            tmp_driver = webdriver.Chrome(chrome_options=tmp_option, service_log_path=os.devnull)
            tmp_driver.get(login_url)
            if tmp_driver.current_url != login_url:
                username = tmp_driver.find_element(By.CSS_SELECTOR, '#ap_email')
                username.clear()
                username.send_keys(username)
                password = tmp_driver.find_element(By.CSS_SELECTOR, '#ap_password')
                password.clear()
                password.send_keys(passwd)
                btn = tmp_driver.find_element(By.CSS_SELECTOR, '#signInSubmit')
                try:
                    btn.click()
                except:
                    self.__write_log('warning 000')
            time.sleep(5)
            tmp_cookies = tmp_driver.get_cookies()
            tmp_driver.quit()

            # dump to pickle
            with open(self.__cookies_file_path, 'wb') as f:
                pickle.dump([cookie for cookie in tmp_cookies if cookie['domain'] == '.amazon.com'], f)

        self.__driver.delete_all_cookies()
        with open(self.__cookies_file_path, 'rb') as f:
            saved_cookies = pickle.load(f)
        for cookie in saved_cookies:
            self.__driver.add_cookie(cookie)

    @staticmethod
    def __get_default_browser_profile(browser_name):
        if browser_name == 'firefox':
            profile_path = '%s/Mozilla/Firefox/Profiles' % os.environ["APPDATA"]
            dirs = os.listdir(profile_path)
            for i in dirs:
                if i.endswith('.default'):
                    return '%s/%s' % (profile_path, i)
        elif browser_name == 'chrome':
            return '%s/Google/Chrome/User Data' % os.environ["LOCALAPPDATA"]
        else:
            assert False, 'Unknown browser_name'
