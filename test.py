def test_str():
    # expected ['Doi song']
    test_doc = '''
        Cô gái 28 tuổi xoay ngang xoay dọc để tìm đường sống. Trong một lần trò chuyện với nhóm xuất khẩu nông sản quen biết từ thời kinh doanh thực phẩm sạch, Hồng nhận được một gợi ý sản xuất và cung cấp mỳ, trong đó có mỳ làm từ ngô để xuất đi các nước châu Âu. Nhu cầu tích trữ đồ khô trong bối cảnh dịch bệnh khiến thị trường bún, mỳ xuất khẩu phát triển mạnh.

        Ngô là thứ lương thực nuôi sống người Nùng ở Hữu Lũng và đưa ba chị em Hồng đến giảng đường đại học. Trong miền ký ức gian khổ mà êm đềm nhất luôn có bóng dáng những bữa cơm độn, cháo ngô, bánh chông chênh hay bỏng ngô.
        
        "Nghe đến từ mỳ ngô, trong tôi như có một tiếng gọi xa xưa vọng lại. Tôi từng mơ ước làm ra một sản phẩm có 'mã vạch' quê hương và tự hỏi liệu đây có phải cơ hội cho mình không", Hồng kể.
        
        Tại châu Âu, dòng mỳ pasta làm từ ngô canh tác tự nhiên là sản phẩm cao cấp, một phần vì trong ngô không có gluten - một chất mà những người theo trường phái ăn uống lành mạnh (healthy) cố tránh. Càng nghiên cứu, Minh Hồng càng thấy mỳ ngô có tương lai. Sau hai tuần lên kế hoạch, cô để chồng con ở lại thành phố, về quê cách đó 100 cây số.
        
        Minh Hồng cùng em gái liên hệ nhập khẩu dây chuyền máy móc từ nước ngoài, xây nhà xưởng, lên kế hoạch trồng ngô cũng như hoàn thiện các thủ tục pháp lý... Trong vụ ngô hè thu năm ngoái, cô kêu gọi họ hàng trồng giống ngô bản địa, không biến đổi gene và sử dụng phân trâu, bò, dê thay thế cho phân bón hóa học và không dùng thuốc diệt cỏ. Riêng gia đình Hồng tập trung xây dựng nhà xưởng sản xuất.
        
        Mỳ, miến vốn là món ăn quen thuộc của người Việt. Gia đình Hồng cũng từng làm bún gạo để ăn. Ban đầu cô cứ nghĩ làm mỳ ngô cũng tương tự nhưng khi bắt tay vào thực tế thì khác hoàn toàn.
        
        Mẻ đầu tiên, ngô không ra sợi, chỗ sống chỗ chín. Bản chất ngô không kết dính được như gạo, cứ bở bùng bục. Hồng nghĩ vấn đề tại bột sống nên mẻ sau cần phải làm sao cho chín. Cô chỉnh lại máy, làm lại bột, lần này thành phẩm là những cục cứng ngoắc. Nhìn 40 kg ngô cho ra thành phẩm không khác gì đá tảng, ông Hoàng Văn Hoa, bố Hồng thở dài: "Ném trâu còn chết".
        
        Cứ thế cải tiến dần dần, từ cục to ra cục bé hơn, rồi ra sợi đứt gãy. Bao ngày đêm cả nhà dồn tâm sức vào, cuốn sổ của Hồng ghi chi chít từng mẻ mà không có dấu hiệu khả quan.
        
        Giữa lúc đó, những tai nạn không ngừng ập đến. Trong một lần ông Hoa đang hàn lại máy ép sợi thì bị vập mu bàn tay vào thanh sắt, máu túa ra. Mấy mẹ con hốt hoảng đưa bố đi cấp cứu.
        
        Minh Hồng vẫn phải vừa lo cho bố, vừa lo mẻ bột còn dang dở. Cô nhờ người em chồng có kinh nghiệm cơ khí đến sửa. Vẫn chiếc máy đó và mối hàn đó, người em bị bập ngón chân vào. Lần này máu đổ nhiều hơn. "Những ngày đó tôi không thể ngủ được. Tôi lo quá tam ba bận, nhỡ có chuyện gì xảy ra nguy hiểm hơn nữa", cô bộc bạch.
        
        Hồng quyết định múc bỏ bột, dừng sản xuất vài tuần để người thân tĩnh dưỡng và cũng để cho bản thân cơ hội trấn an. Khi bố khỏe, cô thuê thêm vài thợ về cùng vận hành lại xưởng. Việc này tưởng đã êm thấm, không ngờ chừng một tháng sau khi đang thao tác máy bón mì tự động, ông Hoa bị tay cầm máy đập vào mắt trái, khiến mắt sưng vù, nước mắt chảy ra liên tục. Lần này ông nằm viện mất cả tháng.
        
        "Tôi thực sự nản. Tôi dằn vặt bởi suy nghĩ vì mình mà người thân cũng lâm vào con đường khó khăn", Hồng bộc bạch.
        
        Nhớ lại thời điểm đó, ông Hoa cũng nhận ra tâm trạng của con gái cả. Sau tai nạn này Hồng bảo bố mẹ dừng một thời gian hãy làm, ông Hoa động viên con: "Cố làm nốt đợt này rồi đợi mẻ ngô mới".
    '''
    return test_doc